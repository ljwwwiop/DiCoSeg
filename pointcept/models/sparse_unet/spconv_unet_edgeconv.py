"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


# from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_add, scatter_mean, scatter_std, scatter_max

import spconv.pytorch as spconv
from torch_geometric.utils import scatter
from torch_geometric.nn import knn_graph


from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

import pdb

## ------



class Fusion_EdgeConv(nn.Module):

    def __init__(self, 
                 feat_dim=256, 
                 text_dim=512, 
                 pos_dims=[32, 32, 32, 32],
                 radius_list=[4.0, 8.0, 16.0, 24.0],
                 k_neighbors=16):
        super().__init__()

        # 图特征提取MLP
        self.graph_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim // 2),  # 输入: [x_i, x_j, d_ij]
            nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
        self.k = k_neighbors

    def forward(self, data):
        """
        x: SparseTensor [M, feat_dim]
        text_feat: [19, 512] 
        coords: [M, 3] (x,y,z)
        """
        # pdb.set_trace()
        x = data
        coords = x.indices[:, 1:].float() # [M, 4]
        batch_ids = x.indices[:, 0].long()

        #######################################
        # 步骤1: 基于Distance构建KNN图
        #######################################
        edge_index = knn_graph(coords, k=self.k, batch=batch_ids)  # [2, M*k]

        #######################################
        # 步骤2: 计算边特征（距离+相对位置）
        #######################################
        src, dst = edge_index
        # rel_coords = coords[dst] - coords[src]  # [M*k, 3]
        # distances = torch.norm(rel_coords, dim=1, keepdim=True)  # [M*k, 1]

        x_feats = x.features  # [M, feat_dim]
        neighbor_feats = x_feats[dst]  # [M*k, feat_dim]
        
        # 拼接中心点特征、邻居特征和距离
        edge_feats = torch.cat([
            x_feats[src], 
            neighbor_feats, 
        ], dim=1)  # [M*k, feat_dim*2+1]

        graph_feats = self.graph_mlp(edge_feats)  # [M*k, feat_dim]

        # 均值池化聚合邻居信息
        mean_aggregated = scatter_mean(
            graph_feats, 
            src, 
            dim=0, 
            dim_size=x_feats.size(0)
        )  # [M, feat_dim]
        
        # Max Pooling聚合邻居信息
        max_aggregated = scatter_max(
            graph_feats,
            src,
            dim=0,
            dim_size=x_feats.size(0)
        )[0]  # scatter_max返回(values, indices)，取values部分 # [M, feat_dim]

        feat = mean_aggregated + max_aggregated
        data = data.replace_feature(feat)
        return data
        


## -----

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


@MODELS.register_module("SpUNet-v1m1-edge")
class SpUNetBaseEdge(nn.Module):
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

        # ----
        radius_list=[2.4, 4.8, 14.4] ## sk
        pos_dims = [32, 64, 128]
        self.cmf1 = Fusion_EdgeConv(feat_dim=256, pos_dims=pos_dims, radius_list=radius_list) 
        self.cmf2 = Fusion_EdgeConv(feat_dim=256, pos_dims=pos_dims, radius_list=radius_list)
        # ----

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
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
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)

        x = self.cmf1(x)
        x = self.cmf2(x)

        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )
        return x.features
