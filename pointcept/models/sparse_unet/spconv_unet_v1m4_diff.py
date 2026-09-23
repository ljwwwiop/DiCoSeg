from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

import torch_scatter

import math
import numpy as np

class CondFusionLocal(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.cond_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(inplace=True)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )

        self.norm = nn.BatchNorm1d(dim)

    def forward(self, noise_feat, cond_feat):

        cond_token = self.cond_proj(cond_feat)
        cond_token = cond_token.expand(-1, -1, noise_feat.size(2))

        fused_feat = torch.cat([noise_feat, cond_token], dim=1)
        fused_feat = self.fusion_mlp(fused_feat)
        fused_feat = self.norm(fused_feat)

        out = fused_feat + noise_feat
        return out

class BasicBlockCondition(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        indice_key=None,
        norm_fn=None,
        bias=False,
    ):
        super().__init__()

        groups = 4
        mlp_ratio = 2
        self.groups = groups

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels * mlp_ratio,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            indice_key=indice_key
        )
        self.bn1 = nn.BatchNorm1d(embed_channels * mlp_ratio)

        self.groups = groups

        self.conv2 = spconv.SubMConv3d(
            embed_channels * mlp_ratio,
            embed_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            indice_key=indice_key
        )

        self.bn2 = nn.BatchNorm1d(embed_channels)

        self.proj = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, embed_channels,
                kernel_size=1, stride=stride, bias=False
            ),
            norm_fn(embed_channels),
        ) if (in_channels != embed_channels or stride != 1) else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))

        features = out.features
        B, C = features.shape[:2]
        features = features.view(B, self.groups, C // self.groups)
        features = features * torch.sigmoid(features.mean(dim=2, keepdim=True))
        out = out.replace_feature(features.reshape(B, C))

        out = self.conv2(out)
        out = out.replace_feature(self.relu(self.bn2(out.features)))

        out = out.replace_feature(
            out.features + self.proj(residual).features
        )

        return out.replace_feature(self.relu(out.features))

class RecDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000, schedule='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps
        assert schedule in ['linear', 'cosine'], "schedule must be 'linear' or 'cosine'"
        self.schedule = schedule

        self.register_buffer('betas', self._make_beta_schedule())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        mask_ratio_schedule = torch.linspace(0.1, 0.9, num_timesteps)
        self.register_buffer('mask_ratio_schedule', mask_ratio_schedule)

    def _make_beta_schedule(self):
        if self.schedule == 'linear':

            beta_start = 1e-4
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif self.schedule == 'cosine':

            steps = self.num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
            return betas
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def q_sample(self, x_start, t, noise=None, use_mask=False):

        if noise is None:
            noise = torch.randn_like(x_start.features)

        batch_indices = x_start.indices[:, 0].long()

        t_expanded = t[batch_indices]

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t_expanded, x_start.features.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t_expanded, x_start.features.shape)

        if use_mask:

            mask_ratio = extract(self.mask_ratio_schedule, t_expanded, x_start.features.shape)
            rand = torch.rand_like(mask_ratio)
            mask = (rand < mask_ratio).float()

            noisy_features = x_start.features * (1 - mask) + \
                            (sqrt_alpha * x_start.features + sqrt_one_minus_alpha * noise) * mask
        else:
            noisy_features = sqrt_alpha * x_start.features + sqrt_one_minus_alpha * noise

        return noisy_features

    def forward(self, x, t=None, noise=None, use_mask=False):

        b = x.batch_size
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=x.features.device)

        x_noisy = self.q_sample(x, t, noise, use_mask)
        return x_noisy, t

def extract(a, t, x_shape):

    out = a[t]

    return out.view(-1, 1)

def swish(x):
    return x * torch.sigmoid(x)

def calc_t_emb(ts, t_emb_dim):

    assert t_emb_dim % 2 == 0

    ts = ts.unsqueeze(1)
    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(ts.device)

    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)

    return t_emb

def sparse_to_dense_batch_with_mapping(cond_feat, batch_indices):

    B = int(batch_indices.max().item()) + 1

    feats_by_batch = []

    max_len = 0

    for b in range(B):
        mask = batch_indices == b
        feats = cond_feat[mask]
        feats_by_batch.append(feats.T)

        max_len = max(max_len, feats.shape[0])

    padded_feats = []
    pad_record = []
    for f in feats_by_batch:
        pad_len = max_len - f.shape[1]
        pad_record.append(f.shape[1])
        if pad_len > 0:
            pad = f.new_zeros(f.shape[0], pad_len)
            f = torch.cat([f, pad], dim=1)
        padded_feats.append(f)

    dense_feat = torch.stack(padded_feats, dim=0)
    return dense_feat, pad_record, batch_indices

def dense_to_sparse_using_mapping(dense_feat, pad_record, batch_indices):

    B, C, P = dense_feat.shape
    recovered = []

    for b in range(B):
        valid_len = pad_record[b]
        f = dense_feat[b, :, :valid_len].T
        recovered.append(f)

    recovered_feat = torch.cat(recovered, dim=0)
    assert recovered_feat.shape[0] == batch_indices.shape[0], "恢复数量和原始稀疏点数不一致"
    return recovered_feat

def sparse_to_voxel_patch_fast(cond_feat, batch_indices, coords, max_patch=2048, voxel_size=0.05, importance_mode="variance"):
    B = int(batch_indices.max().item()) + 1
    C = cond_feat.shape[1]
    device = cond_feat.device

    voxel_feats_list = []
    voxel_coords_list = []
    pad_record = []

    for b in range(B):
        mask = (batch_indices == b)
        feats_b = cond_feat[mask]
        coords_b = coords[mask]

        N_b = feats_b.shape[0]
        if N_b == 0:
            voxel_feats_list.append(torch.zeros(C, max_patch, device=cond_feat.device))
            voxel_coords_list.append(torch.zeros(3, max_patch, device=coords.device))
            pad_record.append(0)
            continue

        voxel_idx = (coords_b / voxel_size).floor().long()

        unique_voxels, inverse_indices = torch.unique(voxel_idx, dim=0, return_inverse=True)

        V = unique_voxels.shape[0]

        voxel_feats = torch_scatter.scatter_mean(feats_b, inverse_indices, dim=0)
        voxel_feats = voxel_feats.T

        voxel_coords_center = (unique_voxels.float() + 0.5) * voxel_size
        voxel_coords_center = voxel_coords_center.T

        if V > max_patch:
            if importance_mode == "norm":

                importance_scores = torch.norm(voxel_feats, p=2, dim=0)
            elif importance_mode == "density":

                point_counts = torch.bincount(inverse_indices, minlength=V).float()
                importance_scores = point_counts
            elif importance_mode == "variance":

                voxel_var = torch_scatter.scatter_std(feats_b, inverse_indices, dim=0)
                importance_scores = torch.norm(voxel_var, p=2, dim=1)
            else:
                indices = torch.linspace(0, V-1, max_patch).long().to(cond_feat.device)

            _, indices = torch.topk(importance_scores, max_patch)
            indices = indices.to(device)

            voxel_feats = voxel_feats[:, indices]
            voxel_coords_center = voxel_coords_center[:, indices]
            pad_record.append(max_patch)
        else:
            pad_len = max_patch - V
            if pad_len > 0:
                pad_feat = torch.zeros(C, pad_len, device=cond_feat.device)
                pad_coord = torch.zeros(3, pad_len, device=coords.device)
                voxel_feats = torch.cat([voxel_feats, pad_feat], dim=1)
                voxel_coords_center = torch.cat([voxel_coords_center, pad_coord], dim=1)
            pad_record.append(V)

        voxel_feats_list.append(voxel_feats)
        voxel_coords_list.append(voxel_coords_center)

    voxel_feats = torch.stack(voxel_feats_list, dim=0)
    voxel_coords = torch.stack(voxel_coords_list, dim=0)

    return voxel_feats, voxel_coords, pad_record, batch_indices

class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:

            self.proj_conv = spconv.SubMConv3d(
                in_channels, embed_channels, kernel_size=1, bias=False
            )
            self.proj_norm = norm_fn(embed_channels)

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        x, condition, context = x
        residual = x

        out = self.conv1(x)

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)

        out = out.replace_feature(self.bn2(out.features))

        if self.in_channels == self.embed_channels:
            residual = self.proj(residual)
        else:
            residual = residual.replace_feature(
                self.proj_norm(self.proj_conv(residual).features)
            )
        out = out.replace_feature(out.features + residual.features)
        out = out.replace_feature(self.relu(out.features))
        return out, condition, context

class SPConvDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        indice_key,
        kernel_size=2,
        bias=False,
        norm_fn=None,
    ):
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)

        out = out.replace_feature(self.bn(out.features))
        out = out.replace_feature(self.relu(out.features))
        return out

class SPConvUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        indice_key,
        kernel_size=2,
        bias=False,
        norm_fn=None,
    ):
        super().__init__()
        self.conv = spconv.SparseInverseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)

        out = out.replace_feature(self.bn(out.features))
        out = out.replace_feature(self.relu(out.features))
        return out

class SPConvPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, norm_fn=None):
        super().__init__()
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
            indice_key="stem",
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)

        out = out.replace_feature(self.bn(out.features))
        out = out.replace_feature(self.relu(out.features))
        return out

@MODELS.register_module("SpUNet-v1m4-diff")
class SpUNetBaseV3DIFF(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        zero_init=True,
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.cls_mode = cls_mode
        self.conditions = conditions
        self.zero_init = zero_init

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = SPConvPatchEmbedding(
            in_channels, base_channels, kernel_size=5, norm_fn=norm_fn
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        self.noise_layer = RecDiffusion(num_timesteps=1000, schedule='linear')

        self.rec_down = nn.ModuleList()
        self.rec_enc = nn.ModuleList()
        self.cond_fusion = nn.ModuleList()
        self.cond_fusion_local = nn.ModuleList()

        diff_in_channels = num_classes
        self.noise_conv_input = SPConvPatchEmbedding(
            diff_in_channels, base_channels, kernel_size=5, norm_fn=norm_fn
        )
        self.eps = 1e-6

        self.time_dim = 128

        self.time_mlp1 = nn.ModuleList()
        self.activation = swish

        for s in range(self.num_stages):

            self.down.append(
                SPConvDown(
                    enc_channels,
                    channels[s],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{s + 1}",
                    norm_fn=norm_fn,
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )

            self.time_mlp1.append(
                nn.Linear(self.time_dim, enc_channels)
            )
            self.rec_down.append(
                SPConvDown(
                    enc_channels,
                    channels[s],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{s + 1}",
                    norm_fn=norm_fn,
                )
            )
            self.rec_enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            (
                                f"rec_block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )

            self.cond_fusion_local.append(
                CondFusionLocal(channels[s])
            )

            self.cond_fusion.append(
                BasicBlockCondition(channels[s]*2, channels[s], indice_key=f'cond_{s+1}', norm_fn=norm_fn)
            )

            if not self.cls_mode:

                self.up.append(
                    SPConvUp(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        norm_fn=norm_fn,
                    )
                )
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        f"block{i}",
                                        block(
                                            dec_channels + enc_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                    if i == 0
                                    else (
                                        f"block{i}",
                                        block(
                                            dec_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = (
            channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )

        self.rec_final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            if m.affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        condition = input_dict["condition"][0] if 'condition' in input_dict.keys() else None
        context = input_dict["context"] if "context" in input_dict.keys() else None

        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )

        d_skips = []
        if self.training:
            segment = input_dict['segment']
            ignore_mask = (segment == -1)

            segment_safe = segment.clone()
            segment_safe[ignore_mask] = 0

            n_x0 = torch.log(torch.nn.functional.one_hot(segment_safe, self.num_classes).float() + self.eps)

            n_x0[ignore_mask] = 0.0

            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
            noise_seg_x = spconv.SparseConvTensor(
                features=n_x0,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1,
            )

            noise_seg_feat, t = self.noise_layer(noise_seg_x)

            noise_x = noise_seg_x.replace_feature(noise_seg_feat)

            t_emb = calc_t_emb(t, self.time_dim)
            noise_x = self.noise_conv_input([noise_x, condition, context])

            d_skips = [noise_x]
            max_patch_list = [2048, 2048, 1024, 512, 256]

        x = self.conv_input([x, condition, context])
        skips = [x]

        diff_losses = 0.

        for s in range(self.num_stages):

            x = self.down[s]([x, condition, context])

            x, _, _ = self.enc[s]([x, condition, context])
            skips.append(x)

            if self.training:
                t_emb_expanded = t_emb[noise_x.indices[:, 0]]
                t_emb_expanded = self.time_mlp1[s](t_emb_expanded)
                t_emb_expanded = self.activation(t_emb_expanded)

                noise_x = noise_x.replace_feature(noise_x.features + t_emb_expanded)

                noise_x = self.rec_down[s]([noise_x, condition, context])
                noise_feat, noise_pad_record, noise_batch_indices = sparse_to_dense_batch_with_mapping(noise_x.features, noise_x.indices[:, 0])

                cond_feat, cond_voxel, pad_record, batch_indices = sparse_to_voxel_patch_fast(x.features, x.indices[:, 0], x.indices[:,1:], max_patch=max_patch_list[s])
                cond_feat = self.cond_fusion_local[s](noise_feat, cond_feat)

                cond_feat = dense_to_sparse_using_mapping(cond_feat, noise_pad_record, noise_batch_indices)
                noise_x = noise_x.replace_feature(cond_feat)

                noise_x = noise_x.replace_feature(torch.cat([noise_x.features, x.features], dim=1))
                noise_x = self.cond_fusion[s](noise_x)

                noise_x, _, _ = self.rec_enc[s]([noise_x, condition, context])
                d_skips.append(noise_x)

        x = skips.pop(-1)
        if len(d_skips) > 0:

            noise_x = d_skips.pop(-1)

        if not self.cls_mode:

            for s in reversed(range(self.num_stages)):
                x = self.up[s]([x, condition, context])
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x, _, _ = self.dec[s]([x, condition, context])

                if self.training:
                    noise_x = self.up[s]([noise_x, condition, context])
                    d_skip = d_skips.pop(-1)
                    noise_x = noise_x.replace_feature(torch.cat((noise_x.features, d_skip.features), dim=1))
                    noise_x, _, _ = self.dec[s]([noise_x, condition, context])

            if self.training:
                rec_noise_x = self.rec_final(noise_x)

                seg_gt_mask = segment[~ignore_mask]
                diff_feat = rec_noise_x.features[~ignore_mask]
                diff_losses = F.cross_entropy(diff_feat, seg_gt_mask)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )

        return x.features, diff_losses
