"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import pdb

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

# ------ DOWNSAMPLE 


class ConsistentSPConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key=None, kernel_size=2):
        super().__init__()
        self.conv = spconv.SparseConv3d(in_channels, out_channels, 
                                      kernel_size=kernel_size,
                                      stride=kernel_size,
                                      indice_key=indice_key)
        self.saliency_head = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 显著性预测
        # sparse-to-dense
        # pdb.set_trace()
        saliency = self.saliency_head(x.features)
        
        # 执行卷积
        out = self.conv(x)
        
        # 显著性传播
        buf = out.find_indice_pair(self.conv.indice_key)
        valid_mask = buf.pair_fwd >= 0
        out_indices, in_indices = torch.where(valid_mask)
        
        # 加权特征增强
        out_saliency_max = scatter(saliency[in_indices], out_indices, 
                             dim_size=out.features.shape[0], reduce="max")
        # out_saliency_mean = scatter(saliency[in_indices], out_indices,
        #                   dim_size=out.features.shape[0], reduce="mean")

        enhanced_feats = out.features * (1 + out_saliency_max)
        
        # 常规处理
        out = out.replace_feature(self.relu(self.bn(enhanced_feats)))
        return out



# ------

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

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

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
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


@MODELS.register_module("SpUNet-downsample")
class SpUNetBaseDS(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
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

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        # ---
        self.channel_adjust = nn.ModuleList()
        self.fusion_weights = nn.ModuleList()

        # ---

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                # spconv.SparseSequential(
                #     spconv.SparseConv3d(
                #         enc_channels,
                #         channels[s],
                #         kernel_size=2,
                #         stride=2,
                #         bias=False,
                #         indice_key=f"spconv{s + 1}",
                #     ),
                #     norm_fn(channels[s]),
                #     nn.ReLU(),
                # )


                ConsistentSPConvDown(
                    enc_channels,
                    channels[s],
                    indice_key=f"spconv{s + 1}",
                    kernel_size=2,
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
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

            # --------
            self.channel_adjust.append(
                    nn.Sequential(
                        nn.Linear(enc_channels, channels[s]),
                        nn.BatchNorm1d(channels[s]),
                        nn.ReLU()
                    )
                )
            # 融合权重
            self.fusion_weights.append(
                nn.Sequential(
                    nn.Linear(channels[s], channels[s]),
                    nn.Sigmoid()  # 自适应权重[0,1]
                )
            )
            # --------

            if not self.cls_mode:
                # decode num_stages
                self.up.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(
                            channels[len(channels) - s - 2],
                            dec_channels,
                            kernel_size=2,
                            bias=False,
                            indice_key=f"spconv{s + 1}",
                        ),
                        norm_fn(dec_channels),
                        nn.ReLU(),
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

        # drop 0 layer
        self.channel_adjust[0] = spconv.Identity()
        self.fusion_weights[0] = spconv.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

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
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            prev_x = x

            x = self.down[s](x)
            x = self.enc[s](x)

            # ---- 跨层特征融合 ----
            if s > 0:  # 从第二层开始融合
                # 1. 坐标对齐（处理下采样导致的点集变化）
                buf = x.find_indice_pair(self.down[s-1].conv.indice_key)
                valid_mask = buf.pair_fwd >= 0
                out_indices, in_indices = torch.where(valid_mask)
                
                # 2. 特征维度匹配
                if prev_x.features.shape[1] != x.features.shape[1]:
                    prev_feats = self.channel_adjust[s](prev_x.features)  # [N,C1]->[N,C2]
                else:
                    prev_feats = prev_x.features
                
                # 3. 特征传播 -> TODO cluster for nearest
                propagated_feats = torch.zeros_like(x.features)
                propagated_feats[out_indices] = prev_feats[in_indices]  # 坐标映射
                
                # 4. 残差融合
                x = x.replace_feature(
                    x.features + self.fusion_weights[s](propagated_feats)  # [M,C2]
                )
            # ---------------

            skips.append(x)
        x = skips.pop(-1)

        # dec forward
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)

        x = self.final(x)

        return x.features

