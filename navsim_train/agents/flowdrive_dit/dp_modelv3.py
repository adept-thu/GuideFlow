# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict, List
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
#from diffusers import DDPMScheduler

from navsim.agents.gtrs_flow.dp_config import DPConfig
from navsim.agents.gtrs_dense.hydra_backbone_bev import HydraBackboneBEV
from navsim.agents.gtrs_flow.rectifiedflow import RectifiedFlow
from navsim.agents.gtrs_flow.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention, pos2posemb2d

x_diff_min = -1.2698211669921875
x_diff_max = 7.475563049316406
x_diff_mean = 2.950225591659546

# Y difference statistics
y_diff_min = -5.012081146240234
y_diff_max = 4.8563690185546875
y_diff_mean = 0.0607292577624321

# Calculate scaling factors for differences
x_diff_scale = abs(x_diff_max - x_diff_min)
y_diff_scale = abs(y_diff_max - y_diff_min)

HORIZON = 8
ACTION_DIM = 4
ACTION_DIM_ORI = 3


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len):
        super().__init__()
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        token_len = obs_len + 1
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                timestep,
                cond):
        #import pdb;pdb.set_trace()
        cond = cond#.detach()
        B, HORIZON, DIM = sample.shape # 12, 8, 4
        sample = sample.view(B, -1).contiguous().float() # [12,8,4] --> [12,32]
        input_emb = self.input_emb(sample) # [12,256]

        timesteps = timestep # [12]
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device) # [12]
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1) # 12,1,256
        # (B,To,n_emb)
        cond_embeddings = torch.cat([time_emb, cond], dim=1) # [12, 66, 256]
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
                              :, :tc, :
                              ]  # each position maps to a (learnable) vector [1, 66, 256]
        # import pdb;pdb.set_trace()
        # [12, 66, 256] + [1, 66, 256]
        x = cond_embeddings + position_embeddings
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb.unsqueeze(1) # [12,1,256]
        t = token_embeddings.shape[1] # 1
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector [1, 66, 256]
        x = token_embeddings + position_embeddings # x.shape = 12,1,256
        # (B,T,n_emb)
        x = self.dp_transformer(
            tgt=x,
            memory=memory, # 12,66,256
        )
        # (B,T,n_emb)
        x = self.ln_f(x) # 12,1,256
        x = self.output_emb(x) # 12, 1, 32
        return x.squeeze(1).view(B, HORIZON, DIM).contiguous() # [12,8,4]


def diff_traj(traj):
    B, L, _ = traj.shape # 12,8,3
    sin = traj[..., -1:].sin() # [12, 8, 1]
    cos = traj[..., -1:].cos() # [12, 8, 1]
    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device) # [12, 1, 1]
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad) # [12,8,1] 制作轨迹的查分
    x_diff = x_diff - x_diff_min
    x_diff_norm = (2* x_diff / x_diff_scale) - 1 # [12,8,1]

    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    y_diff = traj[..., 1:2].diff(n=1, dim=1, prepend=zero_pad)
    y_diff = y_diff - y_diff_min
    y_diff_norm = (2 * y_diff / y_diff_scale) - 1

    return torch.cat([x_diff_norm, y_diff_norm, sin, cos], -1) # [12.8.4]


def cumsum_traj(norm_trajs):
    B, L, _ = norm_trajs.shape
    sin_values = norm_trajs[..., 2:3]
    cos_values = norm_trajs[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)

    # Denormalize x differences
    #x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff = (norm_trajs[..., 0:1] + 1)/2 * x_diff_scale + x_diff_min

    # Denormalize y differences
    #y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff = (norm_trajs[..., 1:2] + 1)/2 * y_diff_scale + y_diff_min

    # Cumulative sum to get absolute positions
    x = x_diff.cumsum(dim=1)
    y = y_diff.cumsum(dim=1)

    return torch.cat([x, y, heading], -1)


def model_fn(x, tim, kv):
    model.eval()
    return model(x, tim, kv)


class DPHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DPConfig = None
                 ):
        super().__init__()
        self.config = config
        img_num = 2 if config.use_back_view else 1

        self.transformer_dp = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, config.dp_layers,
            input_dim=ACTION_DIM * HORIZON,
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
        )
        
        self.rectified_flow = RectifiedFlow(init_type='gaussian', noise_scale=1.0, use_ode_sampler='rk45',sample_N=100)
    
    def get_Z0_Zt_t_target(self, sde, batch, reduce_mean=True, eps=1e-3):
        
        z0 = sde.get_z0(batch).to(batch.device)
        
        
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

        t_expand = t.view(-1, 1, 1, 1).contiguous().repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
        perturbed_data = t_expand * batch + (1.-t_expand) * z0
        target = batch - z0 
         
        return z0, perturbed_data, target, t
    
    
    def generate_condition_mask(self,p_drop,condition_anchor_feature):
        mask = (torch.rand(condition_anchor_feature.shape[0], device=condition_anchor_feature.device) < p_drop)
        mask_expanded = mask.float()  # 转换为 float (True->1.0, False->0.0)
        mask_expanded = mask_expanded.view(-1, 1, 1).contiguous()  # 形状变为 [3, 1, 1]
        result = mask_expanded.expand(condition_anchor_feature.shape)
        return result

    def forward(self, kv) -> Dict[str, torch.Tensor]:
        B = kv.shape[0]
        device = kv.device
        start_time = time.time()
        with torch.no_grad():
            zt = torch.randn(
                B, 8, 4,
                dtype=torch.float32,           
                device=device
            )
            timesteps = [i for i in range(100)]
            dt = 1.0/100
            for i_step in timesteps:
                tim = torch.tensor(i_step, device=device).expand(B)
                v_pred = self.transformer_dp(zt, tim, kv)
                zt = zt + v_pred * dt
            traj = cumsum_traj(zt)
        result = {}
        #import pdb;pdb.set_trace()
        result['trajectory'] = traj
        end_time = time.time()
        print(f"单次推理耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
        return result

    def get_dp_loss(self, kv, gt_trajectory):
        #import pdb;pdb.set_trace() # kv.shape = [12, 65, 256]
        B = kv.shape[0]
        device = kv.device # device(type='cuda', index=0)
        gt_trajectory = gt_trajectory.float()
        gt_trajectory = diff_traj(gt_trajectory)
        z0, zt, target, tim  = self.get_Z0_Zt_t_target(self.rectified_flow, gt_trajectory.unsqueeze(1))
        #import pdb;pdb.set_trace()
        timesteps =  tim # timesteps.shape = [12]
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_dp_input = zt.squeeze(1)

        # Predict the noise residual
        pred = self.transformer_dp(
            noisy_dp_input, # [12, 8, 4]
            timesteps, # [12]
            kv
        )
        return F.mse_loss(pred, target.squeeze(1))


class DPModel(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        self._backbone = HydraBackboneBEV(config)

        kv_len = self._backbone.bev_w * self._backbone.bev_h
        emb_len = kv_len + 1
        if self._config.use_hist_ego_status:
            emb_len += 1
        self._keyval_embedding = nn.Embedding(
            emb_len, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        self.downscale_layer = nn.Linear(self._backbone.img_feat_c, config.tf_d_model)
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
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        if self._config.use_hist_ego_status:
            self._hist_status_encoding = nn.Linear((2 + 2 + 3) * 3, config.tf_d_model)

        self._trajectory_head = DPHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        if self._config.use_temporal_bev_kv:
            self.temporal_bev_fusion = nn.Conv2d(
                config.tf_d_model * 2,
                config.tf_d_model,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, features: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                tokens: List[str],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        camera_feature_back: torch.Tensor = features["camera_feature_back"]
        status_feature: torch.Tensor = features["status_feature"][0]
        #import pdb;pdb.set_trace()

        batch_size = status_feature.shape[0]
        assert (camera_feature[-1].shape[0] == batch_size)

        camera_feature_curr = camera_feature[-1]
        if isinstance(camera_feature_back, list):
            camera_feature_back_curr = camera_feature_back[-1]
        else:
            camera_feature_back_curr = camera_feature_back
        img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_curr, camera_feature_back_curr)
        keyval = self.downscale_layer(bev_tokens)
        assert not self._config.use_temporal_bev_kv
        if self._config.use_temporal_bev_kv:
            with torch.no_grad():
                camera_feature_prev = camera_feature[-2]
                camera_feature_back_prev = camera_feature_back[-2]
                img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_prev, camera_feature_back_prev)
                keyval_prev = self.downscale_layer(bev_tokens)
            # grad for fusion layer
            C = keyval.shape[-1]
            keyval = self.temporal_bev_fusion(
                torch.cat([
                    keyval.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w).contiguous(),
                    keyval_prev.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w).contiguous()
                ], 1)
            ).view(batch_size, C, -1).permute(0, 2, 1).contiguous()

        bev_semantic_map = self._bev_semantic_head(up_bev)
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        if self._config.use_hist_ego_status:
            hist_status_encoding = self._hist_status_encoding(features['hist_status_feature'])
            keyval = torch.concatenate([keyval, hist_status_encoding[:, None]], dim=1)

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval)

        output.update(trajectory)

        output['env_kv'] = keyval
        output['bev_semantic_map'] = bev_semantic_map

        return output
