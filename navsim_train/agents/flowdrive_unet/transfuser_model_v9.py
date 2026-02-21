from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from navsim.agents.flowdrive.transfuser_config import TransfuserConfig
from navsim.agents.flowdrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.flowdrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
#from diffusers.schedulers import DDIMScheduler
import pickle
from scipy import ndimage
from navsim.agents.flowdrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.flowdrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention, pos2posemb2d
from navsim.agents.flowdrive.modules.multimodal_loss import LossComputer
from navsim.agents.flowdrive.unet import UNetModelEP as UNetModel
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.flowdrive.rectifiedflow import RectifiedFlow

class V9TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = ReflowTrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, tokens=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        #import pdb;pdb.set_trace()
        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None], targets=targets, global_img=None, tokens=tokens)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output
    
    def forward_test(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, tokens=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        #import pdb;pdb.set_trace()
        trajectory = self._trajectory_head.forward_test(trajectory_query,agents_query, cross_bev_feature, bev_spatial_shape, status_encoding[:, None], targets=targets, global_img=None, tokens=tokens)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)
        

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class ReflowTrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, plan_anchor_path: str, config: TransfuserConfig):
        super(ReflowTrajectoryHead, self).__init__()
        self.num_poses = num_poses
        self.ego_fut_mode = 20
        self.num_inference_steps = 50
        self.model = UNetModel(image_size=(3, 1, 8), in_channels=3, model_channels=32, out_channels=3, num_res_blocks=1, attention_resolutions="16", num_classes=10)
        self.do_classifier_free_guidance = True
        self.num_channels_latents = 2
        condition_plan_anchor = np.load(plan_anchor_path)
        self.condition_plan_anchor = condition_plan_anchor
        self.rectified_flow = RectifiedFlow(init_type='gaussian', noise_scale=1.0, use_ode_sampler='rk45',sample_N=100)
        self.use_target = True
        if self.use_target:
            self.target_point_encoder = nn.Sequential(
                *linear_relu_ln(128, 1, 1,128),
                nn.Linear(128, 128),
            )
        else:
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(128, 1, 1,128),
                nn.Linear(128, 128),
            )
        self.use_guidance_limit = True
        ego_pkl_path = '/nas/users/perception-users/liulin/workspace/DiffusionDrive/assets/ego_data.pkl'
        with open(ego_pkl_path, 'rb') as f:
            self.ego_data = pickle.load(f)
        self.use_condition_limit = True 

        if self.use_condition_limit:
            self.anchor_path = "/nas/users/perception-users/yuguanyi/playground_2506/code/GTRS/exp/test_gtrs_dense_navhard_two_stage/navhard_two_stage.pkl"
            with open(self.anchor_path, 'rb') as f:
                self.anchor = pickle.load(f)

    def get_Z0_Zt_t_target(self, sde, batch, reduce_mean=True, eps=1e-3):
        #import pdb;pdb.set_trace()
        if sde.reflow_flag:
            z0 = batch[0]
            data = batch[1]
            batch = data.detach().clone()
        else:
            z0 = sde.get_z0(batch).to(batch.device)
        
        if sde.reflow_flag:
            if sde.reflow_t_schedule=='t0': ### distill for t = 0 (k=1)
                t = torch.zeros(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            elif sde.reflow_t_schedule=='t1': ### reverse distill for t=1 (fast embedding)
                t = torch.ones(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            elif sde.reflow_t_schedule=='uniform': ### train new rectified flow with reflow
                t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            elif type(sde.reflow_t_schedule)==int: ### k > 1 distillation
                t = torch.randint(0, sde.reflow_t_schedule, (batch.shape[0], ), device=batch.device) * (sde.T - eps) / sde.reflow_t_schedule + eps
            else:
                assert False, 'Not implemented'
        else:
            ### standard rectified flow loss
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

        t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
        perturbed_data = t_expand * batch + (1.-t_expand) * z0
        target = batch - z0 
        #model_fn = mutils.get_model_fn(model, train=train)
        #score = model_fn(perturbed_data, t*999) ### Copy from models/utils.py 
        return z0, perturbed_data, target, t

    def match_trajectories_fast_old(self, query_traj, candidate_trajs):
        # 扩展 query_traj 以匹配 candidates 的维度
        #import pdb;pdb.set_trace()
        query = query_traj.squeeze(1)  # [64, 8, 2]
        candidates = candidate_trajs  # [256, 8, 2]
        
        # 2. 计算所有pairwise的距离矩阵（使用欧氏距离）
        # query: [64, 8, 2] -> [64, 1, 8, 2]
        # candidates: [256, 8, 2] -> [1, 256, 8, 2]
        diff = query.unsqueeze(1) - candidates.unsqueeze(0)  # [64, 256, 8, 2]
        distances = torch.norm(diff, p=2, dim=-1).mean(dim=-1)  # [64, 256] (对8个点取平均距离)
        
        # 3. 找到每个query的最小距离索引
        _, min_indices = torch.min(distances, dim=1)  # [64]
        
        # 4. 根据索引提取匹配的轨迹
        #min_indices[0] = 255
        #import pdb;pdb.set_trace()
        matched_traj = candidate_trajs[min_indices]
        #matched_traj[0] = 0
        return matched_traj.unsqueeze(1)
    
    
    def match_trajectories_fast(self, query_traj, candidate_trajs, unorm_candidate_trajs):
        # 扩展 query_traj 以匹配 candidates 的维度
        #import pdb;pdb.set_trace()
        query = query_traj.squeeze(1)  # [64, 8, 2]
        candidates = candidate_trajs  # [256, 8, 2]
        query = query[:,:,:2]
        candidates = candidates[:,:,:2]
        angle_query = query_traj.squeeze(1)[:,:,2:]
        angle_candidates = candidate_trajs[:,:,2:]
        #import pdb;pdb.set_trace()
        
        # 2. 计算所有pairwise的距离矩阵（使用欧氏距离）
        # query: [64, 8, 2] -> [64, 1, 8, 2]
        # candidates: [256, 8, 2] -> [1, 256, 8, 2]
        diff = query.unsqueeze(1) - candidates.unsqueeze(0)  # [64, 256, 8, 2]
        angle_diff = angle_query.unsqueeze(1) - angle_candidates.unsqueeze(0)
        #import pdb;pdb.set_trace()
        #distances = torch.norm(diff, p=2, dim=-1).mean(dim=-1)  # [64, 256] (对8个点取平均距离)
        distances = torch.norm(diff, p=2, dim=-1).sum(dim=-1)
        #distances = torch.norm(diff, p=2, dim=-1)[:,:,7]
        angle_distances = torch.norm(angle_diff, p=2, dim=-1).sum(dim=-1)
        sum_distances = 0.7 * distances + (1 - 0.7)*angle_distances
        #import pdb;pdb.set_trace()
        # 3. 找到每个query的最小距离索引
        _, min_indices = torch.min(sum_distances, dim=1)  # [64]
        
        # 4. 根据索引提取匹配的轨迹
        #min_indices[0] = 78
        #import pdb;pdb.set_trace()
        matched_traj = unorm_candidate_trajs[min_indices]
        #matched_traj[0] = 0
        return matched_traj.unsqueeze(1)


    def compute_loss(self, v_pred, v_target):
        
        loss = F.mse_loss(v_pred, v_target)

        return loss
    
    def norm_odo_anchor(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.57)/66.73 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 19.68)/42.26 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 1.80)/3.80 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.5742)/66.7413 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 19.68)/45.265 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 1.80)/3.816 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 66.7413 - 1.5742
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 45.265 - 19.68
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.816 - 1.80
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    
    def generate_condition_mask(self,p_drop,condition_anchor_feature):
        """
        用于控制anchor条件信息对于model的影响
        """
        mask = (torch.rand(condition_anchor_feature.shape[0], device=condition_anchor_feature.device) < p_drop)
        mask_expanded = mask.float()  # 转换为 float (True->1.0, False->0.0)
        mask_expanded = mask_expanded.view(-1, 1, 1)  # 形状变为 [3, 1, 1]
        result = mask_expanded.expand(condition_anchor_feature.shape)
        return result
    
    def generate_ep_mask(self, p_drop, ep_score):
        mask = (torch.rand(ep_score.shape[0], device=ep_score.device) < p_drop)
        result = mask.float()
        return result

    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None, global_img=None, tokens=None) -> Dict[str, torch.Tensor]:
        gt_trajectory = targets['trajectory']
        gt_trajectory = self.norm_odo(gt_trajectory)
        
        z0, zt, target, tim  = self.get_Z0_Zt_t_target(self.rectified_flow, gt_trajectory.permute(0, 2, 1).unsqueeze(2))
        tim = tim * 100
        ep_score = []
        for i_token in tokens:
            ep_score.append(self.ego_data[i_token])
        #import pdb;pdb.set_trace()
        ep_score = torch.tensor(ep_score)
        if self.use_target:
            target_point_feature = targets['trajectory'][:,:,:2][:,7:8,:]
            target_point_feature = pos2posemb2d(target_point_feature)
            target_point_feature = target_point_feature.float()
            condition_anchor_feature = self.target_point_encoder(target_point_feature)
        else:
            plan_anchors_torch = torch.from_numpy(self.condition_plan_anchor).to(gt_trajectory.device)
            norm_gt_trajectory = self.norm_odo_anchor(targets['trajectory'])
            norm_plan_anchors_torch = self.norm_odo_anchor(plan_anchors_torch)
            condition_anchor = self.match_trajectories_fast(norm_gt_trajectory.unsqueeze(1), norm_plan_anchors_torch, plan_anchors_torch)
            condition_anchor = condition_anchor[:,:,:,:2]
            traj_pos_embed = gen_sineembed_for_position(condition_anchor,hidden_dim=16)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_pos_embed = traj_pos_embed.float()
            condition_anchor_feature = self.plan_anchor_encoder(traj_pos_embed) # 64 1, 128
        #timesteps = [(i/25) * 25. for i in range(25)]
        timesteps = [(i/100) * 100. for i in range(100)]
        output = {}
        if self.training:
            condition_anchor_feature = condition_anchor_feature * self.generate_condition_mask(0.80, condition_anchor_feature).to(condition_anchor_feature.device)
            #import pdb;pdb.set_trace()
            ep_score = ep_score.to(tim.device)
            ep_score = ep_score * self.generate_ep_mask(0.80, ep_score).to(tim.device)
            v_pred = self.model(tim, zt, condition_anchor_feature, bev_feature, agents_query, ego_query, ep_score)
            output["trajectory_loss"] = self.compute_loss(v_pred, target.float())
        else:
            #import pdb;pdb.set_trace()
            condition_anchor_feature = condition_anchor_feature * self.generate_condition_mask(0.80, condition_anchor_feature).to(condition_anchor_feature.device)
            ep_score = ep_score.to(tim.device)
            ep_score = ep_score * self.generate_ep_mask(0.80, ep_score).to(tim.device)
            v_pred = self.model(tim, zt, condition_anchor_feature, bev_feature, agents_query, ego_query, ep_score)
            output["trajectory_loss"] = self.compute_loss(v_pred, target.float())

        return output
    
    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None, tokens=None) -> Dict[str, torch.Tensor]:
    
        output = {}
        ep_score = []
        for i_token in tokens:
            ep_score.append(1.0)
        #import pdb;pdb.set_trace()
        ep_score = torch.tensor(ep_score)
        if self.training:
            if self.use_target:
                target_point_feature = targets['trajectory'][:,:,:2][:,7:8,:]
                target_point_feature = pos2posemb2d(target_point_feature)
                target_point_feature = target_point_feature.float()
                condition_anchor_feature = self.target_point_encoder(target_point_feature)
            else:
                plan_anchors_torch = torch.from_numpy(self.condition_plan_anchor).to(gt_trajectory.device)
                norm_gt_trajectory = self.norm_odo_anchor(targets['trajectory'])
                norm_plan_anchors_torch = self.norm_odo_anchor(plan_anchors_torch)
                condition_anchor = self.match_trajectories_fast(norm_gt_trajectory.unsqueeze(1), norm_plan_anchors_torch, plan_anchors_torch)
                condition_anchor = condition_anchor[:,:,:,:2]
                traj_pos_embed = gen_sineembed_for_position(condition_anchor,hidden_dim=16)
                traj_pos_embed = traj_pos_embed.flatten(-2)
                traj_pos_embed = traj_pos_embed.float()
                condition_anchor_feature = self.plan_anchor_encoder(traj_pos_embed) # 64 1, 128
                condition_anchor_feature = condition_anchor_feature * self.generate_condition_mask(0.85, condition_anchor_feature).to(condition_anchor_feature.device)
            gt_trajectory = targets['trajectory']
            gt_trajectory = self.norm_odo(gt_trajectory)
            z0, zt, target, tim  = self.get_Z0_Zt_t_target(self.rectified_flow, gt_trajectory.permute(0, 2, 1).unsqueeze(2))
            tim = tim * 100
            v_pred = self.model(tim, zt, condition_anchor_feature, bev_feature, agents_query, ego_query)
            output["trajectory_loss"]  = self.compute_loss(v_pred,target.float())
        else:
            if self.use_condition_limit:
                target_anchors = []
                for i_token in tokens:
                    limited_anchor = self.anchor[i_token]['trajectory'].poses #[40, 3]
                    sampled_limited_points = [limited_anchor[i] for i in range(4, 40, 5)]
                    sampled_limited_points = np.stack(sampled_limited_points, axis=0) # [8,3]
                    sampled_limited_points = torch.tensor(sampled_limited_points, device=ego_query.device)
                    sampled_limited_points = sampled_limited_points.unsqueeze(0)
                    target_anchors.append(sampled_limited_points)
                target_anchors = torch.cat(target_anchors)
                if self.use_target:
                    target_point_feature = target_anchors[:,:,:2][:,7:8,:]
                    target_point_feature = pos2posemb2d(target_point_feature)
                    target_point_feature = target_point_feature.float()
                    condition_anchor_feature = self.target_point_encoder(target_point_feature)
                zt = torch.randn(
                    ego_query.shape[0], 3, 1, 8,  # Shape
                    dtype=torch.float32,           # Data type
                    device=condition_anchor_feature.device      # Corrected `.device`
                ) # [bs,3,1,8]
            with torch.no_grad():
                timesteps = [(i/100) * 100. for i in range(101)]
                dt = 1.0/100
                for i_step in timesteps:
                    tim = torch.tensor(i_step, device=condition_anchor_feature.device).expand(zt.shape[0])
                    ep_score = ep_score.to(tim.device)
                    if self.use_guidance_limit:
                        if i_step < 10 or i_step >= 90:
                            v_pred_no_condition = self.model(tim, zt, torch.zeros(condition_anchor_feature.shape).to(zt.device), bev_feature, agents_query, ego_query, torch.zeros(ep_score.shape).to(zt.device))
                            v_pred_condition = self.model(tim, zt, torch.zeros(condition_anchor_feature.shape).to(zt.device), bev_feature, agents_query, ego_query, ep_score)
                            v_pred = v_pred_no_condition + 2.5 * (v_pred_condition - v_pred_no_condition)
                        else:
                            v_pred_condition = self.model(tim, zt, condition_anchor_feature, bev_feature, agents_query, ego_query, ep_score)
                            v_pred_no_condition = self.model(tim, zt, torch.zeros(condition_anchor_feature.shape).to(zt.device), bev_feature, agents_query, ego_query, ep_score)
                            v_pred = v_pred_no_condition + 2.5 * (v_pred_condition - v_pred_no_condition)
                    else:
                        v_pred_condition = self.model(tim, zt, condition_anchor_feature, bev_feature, agents_query, ego_query, ep_score)
                        v_pred_no_condition = self.model(tim, zt, torch.zeros(condition_anchor_feature.shape).to(zt.device), bev_feature, agents_query, ego_query, ep_score)
                        v_pred = v_pred_no_condition + 2.5 * (v_pred_condition - v_pred_no_condition)
                    zt = zt + v_pred * dt # [1,2,1,8]
                output['trajectory'] = self.denorm_odo(zt.permute(0,2,3,1).squeeze(0))
        
        return output