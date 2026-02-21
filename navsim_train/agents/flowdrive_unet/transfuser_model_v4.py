# this code is consist of  mean_flow, which is for traj gen--liulin
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange
from functools import partial
from navsim.agents.flowdrive.transfuser_config import TransfuserConfig
from navsim.agents.flowdrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.flowdrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.flowdrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
from navsim.agents.flowdrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention, pos2posemb2d
from navsim.agents.flowdrive.modules.multimodal_loss import LossComputer
from navsim.agents.flowdrive.unet import UNetModel, UNetModelMeanFlow
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.flowdrive.rectifiedflow import RectifiedFlow
from navsim.agents.flowdrive.meanflow import MeanFlow

class V4TransfuserModel(nn.Module):
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

        self._trajectory_head = MeanFlowTrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
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
        bev_feature = bev_feature.permute(0, 2, 1).contiguous()
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1]).contiguous()
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1).contiguous())
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        #import pdb;pdb.set_trace()
        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=None)
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


class MeanFlowTrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, plan_anchor_path: str, config: TransfuserConfig):
        super(MeanFlowTrajectoryHead, self).__init__()
        self.num_poses = num_poses
        self.model = UNetModelMeanFlow(image_size=(2,1,8), in_channels=2, model_channels=32, out_channels=2, num_res_blocks=1, attention_resolutions="16", num_classes=10)
        self.do_classifier_free_guidance = True
        condition_plan_anchor = np.load(plan_anchor_path)
        self.condition_plan_anchor = condition_plan_anchor
        self.mean_flow = MeanFlow(channels=3, image_size=32, num_classes=10, flow_ratio=0.50, time_dist=['lognorm', -0.4, 1.0], cfg_ratio=0.10, cfg_scale=2.0, cfg_uncond='u')
        self.time_dist = ['lognorm', -0.4, 1.0]
        self.cfg_ratio = 0.10
        self.w = 2.0
        self.cfg_uncond = 'u'
        '''
        self.use_target = False
        if self.use_target:
            self.target_point_encoder = nn.Sequential(
                *linear_relu_ln(128,1,1,128),
                nn.Linear(128,128),
            )
        else:
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(128,1,1,128),
                nn.Linear(128,128),
            )
        '''
        self.create_graph = True # if jvp = autograd
    def match_trajectories_fast(self, query_traj, candidate_trajs, unorm_candidate_trajs):
        query = query_traj.squeeze(1)  # [64, 8, 2]
        candidates = candidate_trajs  # [256, 8, 2]
        query = query[:,:,:2]
        candidates = candidates[:,:,:2]
        angle_query = query_traj.squeeze(1)[:,:,2:]
        angle_candidates = candidate_trajs[:,:,2:]
        diff = query.unsqueeze(1) - candidates.unsqueeze(0)  # [64, 256, 8, 2]
        angle_diff = angle_query.unsqueeze(1) - angle_candidates.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=-1).sum(dim=-1)
        angle_distances = torch.norm(angle_diff, p=2, dim=-1).sum(dim=-1)
        sum_distances = 0.7 * distances + (1 - 0.7)*angle_distances
        _, min_indices = torch.min(sum_distances, dim=1)  # [64]
        
        matched_traj = unorm_candidate_trajs[min_indices]
        return matched_traj.unsqueeze(1)
    
    
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

        odo_info_fut_x = 2*(odo_info_fut_x + 1.57)/66.73 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 19.68)/42.26 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 66.73 - 1.57
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 42.26 - 19.68
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def generate_condition_mask(self,p_drop,condition_anchor_feature):
        """
        用于控制anchor条件信息对于model的影响
        """
        mask = (torch.rand(condition_anchor_feature.shape[0], device=condition_anchor_feature.device) < p_drop)
        mask_expanded = mask.float()  # 转换为 float (True->1.0, False->0.0)
        mask_expanded = mask_expanded.view(-1, 1, 1)  # 形状变为 [3, 1, 1]
        result = mask_expanded.expand(condition_anchor_feature.shape)
        return result
    
    def stopgrad(self, x):
        return x.detach()


    def adaptive_l2_loss(self, error, gamma=0.5, c=1e-3):
        """
        Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
        Args:
            error: Tensor of shape (B, C, W, H)
            gamma: Power used in original ||Δ||^{2γ} loss
            c: Small constant for stability
        Returns:
            Scalar loss
        """
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
        p = 1.0 - gamma
        w = 1.0 / (delta_sq + c).pow(p)
        loss = delta_sq  # ||Δ||^2
        return (self.stopgrad(w) * loss).mean()
    
    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        unshape_gt_trajectory = targets['trajectory'][:,:,:2]
        batch_size = unshape_gt_trajectory.shape[0]
        gt_trajectory = self.norm_odo(unshape_gt_trajectory)
        gt_trajectory = gt_trajectory.unsqueeze(1).permute(0,3,1,2).contiguous()
        #import pdb;pdb.set_trace()
        t, r = self.mean_flow.sample_t_r(batch_size, gt_trajectory.device)
        #import pdb;pdb.set_trace()
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone() #64, 2, 1, 8
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
        #t_ = t.detach().clone()
        #r_ = r.detach().clone()
        #import pdb;pdb.set_trace()
        e = torch.randn_like(gt_trajectory)
        z = (1 - t_) * gt_trajectory + t_ * e
        v = e - gt_trajectory
        '''
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
        if self.w is not None:
            with torch.set_grad_enabled(False):
                u_t = self.model(z, t, t, torch.zeros(condition_anchor_feature.shape).to(z.device), bev_feature, agents_query, ego_query)#.detach()
            v_hat = self.w * v + (1 - self.w) * u_t
            if self.cfg_uncond == 'v':
                pass
        else:
        '''
        v_hat = v
        #import pdb;pdb.set_trace()
        #model_partial = partial(self.model, y=condition_anchor_feature, bev_feature=bev_feature, agent_querys=agents_query, ego_query=ego_query)
        model_partial = partial(self.model, bev_feature=bev_feature, agent_querys=agents_query, ego_query=ego_query)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.mean_flow.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.mean_flow.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - self.stopgrad(u_tgt)
        loss = self.adaptive_l2_loss(error)
        output = {}
        output["trajectory_loss"] = loss
        #'''
        if not self.training:
            sample_steps = 2
            z_infer = torch.randn(z.shape).to(gt_trajectory.device)

            t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=gt_trajectory.device)

            for i in range(sample_steps):
                t = torch.full((z_infer.size(0),), t_vals[i], device=gt_trajectory.device)
                r = torch.full((z_infer.size(0),), t_vals[i + 1], device=gt_trajectory.device)

                # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

                t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
                r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

                v = self.model(z_infer, t, r, bev_feature=bev_feature, agent_querys=agents_query, ego_query=ego_query)
                z_infer = z_infer - (t_-r_) * v
            z_infer = z_infer.permute(0,2,3,1).squeeze(0)
            #import pdb;pdb.set_trace()
            z_infer = self.denorm_odo(z_infer)
            output['trajectory'] = z_infer
        #'''
        return output 