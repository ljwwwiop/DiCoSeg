"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import pdb
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

def swish(x):
    return x * torch.sigmoid(x)

# ---- ScaleLong ----
def universal_scalling(s_feat,s_factor=2**(-0.5)):
    return s_feat * s_factor

def exponentially_scalling(s_feat,k=0.8,i=1):
    return s_feat * k**(i - 1)
# ---- ScaleLong ----

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # 自回归掩码
            # mask = torch.tril(torch.ones(K, K, device=q.device)).bool()  # 下三角掩码

            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            # attn = attn.masked_fill(~mask, -1e9)  # 应用自回归掩码

            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,

        T_dim=-1
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.T_dim = T_dim

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        if(self.T_dim != -1):
            self.t_mlp = nn.Linear(T_dim, channels)

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        # pdb.set_trace()
        if (self.T_dim != -1 and "t_emb" in point.keys()):
            t_emb = point['t_emb']
            t_emb = self.t_mlp(t_emb)
            # t_embed + x, (N,32)
            point.feat = shortcut + t_emb
            shortcut = point.feat

        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster

        T_dim=-1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T_dim = T_dim

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        if(self.T_dim == -1):
            # collect information
            point_dict = Dict(
                feat=torch_scatter.segment_csr(
                    self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
                ),
                coord=torch_scatter.segment_csr(
                    point.coord[indices], idx_ptr, reduce="mean"
                ),
                grid_coord=point.grid_coord[head_indices] >> pooling_depth,
                serialized_code=code,
                serialized_order=order,
                serialized_inverse=inverse,
                serialized_depth=point.serialized_depth - pooling_depth,
                batch=point.batch[head_indices],
            )
        else:
            # collect information
            point_dict = Dict(
                feat=torch_scatter.segment_csr(
                    self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
                ),
                coord=torch_scatter.segment_csr(
                    point.coord[indices], idx_ptr, reduce="mean"
                ),
                grid_coord=point.grid_coord[head_indices] >> pooling_depth,
                serialized_code=code,
                serialized_order=order,
                serialized_inverse=inverse,
                serialized_depth=point.serialized_depth - pooling_depth,
                batch=point.batch[head_indices],
                t_emb=point.t_emb[head_indices],
            )


        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster

        skip_connection_mode="add",
        b=1.0,
        s=1.0,
        skip_connection_scale=False,
        skip_connection_scale_i=False
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        self.skip_connection_mode = skip_connection_mode
        self.b = b
        self.s = s
        self.skip_connection_scale = skip_connection_scale
        self.skip_connection_scale_i = skip_connection_scale_i
        if(skip_connection_mode == "cat"):
            self.proj_cat = PointSequential(nn.Linear(out_channels * 2, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent) ## skip
        parent.feat = parent.feat + point.feat[inverse]

        # Scaling Skip Connection Features
        if (self.skip_connection_scale):
            parent.feat = universal_scalling(parent.feat)
        if(self.skip_connection_scale_i is not None):
            parent.feat = exponentially_scalling(parent.feat,i=self.skip_connection_scale_i)

        # FreeU
        if(self.b != 1 or self.s != 1):
            point.feat, parent.feat = freeU(
                point.feat.permute(1, 0).unsqueeze(0),
                parent.feat.permute(1, 0).unsqueeze(0),
                self.b,
                self.s
            )

        # Skip Connection Mode
        if (self.skip_connection_mode == "add"):
            parent.feat = parent.feat + point.feat[inverse]
        elif(self.skip_connection_mode == "cat"):
            parent.feat = self.proj_cat(torch.cat([parent.feat, point.feat[inverse]], dim=-1))

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

###

class SerializedCrossAttention(PointModule):
    def __init__(
        self,
        q_channels,
        kv_channels,
        num_heads,
        q_patch_size,
        kv_patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert q_channels % num_heads == 0
        assert kv_channels % num_heads == 0
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.num_heads = num_heads
        self.scale = qk_scale or (q_channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                    enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                    upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                    upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.q_patch_size = q_patch_size
            self.kv_patch_size = kv_patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.q_patch_size_max = q_patch_size
            self.kv_patch_size_max = kv_patch_size
            self.q_patch_size = 0
            self.kv_patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.q = torch.nn.Linear(q_channels, q_channels, bias=qkv_bias)
        self.kv = torch.nn.Linear(kv_channels, q_channels * 2, bias=qkv_bias)
        self.proj = torch.nn.Linear(q_channels, q_channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(q_patch_size, num_heads) if self.enable_rpe else None
    
    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.q_patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                    torch.div(
                        bincount + self.q_patch_size - 1,
                        self.q_patch_size,
                        rounding_mode="trunc",
                    )
                    * self.q_patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.q_patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                    _offset_pad[i + 1]
                    - self.q_patch_size
                    + (bincount[i] % self.q_patch_size): _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.q_patch_size
                        + (bincount[i] % self.q_patch_size): _offset_pad[i + 1]
                                                           - self.q_patch_size
                        ]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.q_patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, q_point, kv_point):
        if not self.enable_flash:  # True
            self.q_patch_size = min(
                offset2bincount(q_point.offset).min().tolist(), self.q_patch_size_max
            )

        if not self.enable_flash:  # True
            self.kv_patch_size = min(
                offset2bincount(kv_point.offset).min().tolist(), self.kv_patch_size_max
            )

        H = self.num_heads
        q_K = self.q_patch_size
        kv_K = self.kv_patch_size
        C = self.q_channels
        # 这是为了填补batch size，以至于可以并行执行attention。填补所有batch至最大batch的点数量
        q_pad, q_unpad, q_cu_seqlens = self.get_padding_and_inverse(q_point)
        q_order = q_point.serialized_order[self.order_index][q_pad]
        q_inverse = q_unpad[q_point.serialized_inverse[self.order_index]]

        # kv_pad, kv_unpad, kv_cu_seqlens = self.get_padding_and_inverse(kv_point)
        kv_pad, kv_unpad, kv_cu_seqlens = q_pad, q_unpad, q_cu_seqlens
        kv_order = kv_point.serialized_order[self.order_index][kv_pad]
        # kv_inverse = kv_unpad[kv_point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        q = self.q(q_point.feat)[q_order]  # (N,C)
        kv = self.kv(kv_point.feat)[kv_order]  # (N,C * 2)

        # 使用flash attebtion
        if not self.enable_flash:  # False
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q = q.reshape(-1, q_K, H, C // H).permute(0, 2, 1, 3)
            k, v = (
                # [3, patch_size, 3, head, head_dim] -> [3]
                kv.reshape(-1, kv_K, 2, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(q_point, q_order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(q.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_kvpacked_func(
                q=q.half().reshape(-1, H, C // H),
                kv=kv.half().reshape(-1, 2, H, C // H),
                cu_seqlens_q=q_cu_seqlens,
                cu_seqlens_k=kv_cu_seqlens,
                max_seqlen_q=self.q_patch_size,
                max_seqlen_k=self.kv_patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            )
            feat = feat.view(-1, C)
        feat = feat[q_inverse].float()  # return initial point. each index is unque.

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        q_point.feat = feat
        return q_point

class CrossBlock(PointModule):
    def __init__(
        self,
        q_channels,
        kv_channels,
        num_heads,
        q_patch_size=48,
        kv_patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        q_cpe_indice_key=None,
        kv_cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,

        tm_feat=1.0,
        tm_restomer=False
    ):
        super().__init__()
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.pre_norm = pre_norm

        # ---- feature fusion ----
        self.tm_feat = tm_feat
        if (self.tm_feat == "channel_scale"):
            self.feat_scale = nn.parameter.Parameter(torch.full(fill_value=1.0, size=(1, q_channels)))
        elif (self.tm_feat == "b_channel_scale"):
            self.feat_scale = nn.parameter.Parameter(torch.full(fill_value=0.5, size=(1, q_channels)))
        elif (self.tm_feat == "lr_scale"):
            self.feat_scale = nn.parameter.Parameter(torch.full(fill_value=1.0, size=(1,)))
        elif (self.tm_feat == "b_lr_scale"):
            self.feat_scale = nn.parameter.Parameter(torch.full(fill_value=0.5, size=(1,)))
        else:
            self.feat_scale = self.tm_feat
        # ---- feature fusion ----

        self.q_cpe = PointSequential(
            spconv.SubMConv3d(
                q_channels,
                q_channels,
                kernel_size=3,
                bias=True,
                indice_key=q_cpe_indice_key,
            ),
            nn.Linear(q_channels, q_channels),
            norm_layer(q_channels),
        )

        self.kv_cpe = PointSequential(
            spconv.SubMConv3d(
                kv_channels,
                kv_channels,
                kernel_size=3,
                bias=True,
                indice_key=kv_cpe_indice_key,
            ),
            nn.Linear(kv_channels, kv_channels),
            norm_layer(kv_channels),
        )

        self.q_norm1 = PointSequential(norm_layer(q_channels))
        self.kv_norm1 = PointSequential(norm_layer(kv_channels))

        self.attn = SerializedCrossAttention(
            q_channels=q_channels,
            kv_channels=kv_channels,
            num_heads=num_heads,
            q_patch_size=q_patch_size,
            kv_patch_size=kv_patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.q_norm2 = PointSequential(norm_layer(q_channels))

        self.mlp = PointSequential(
            MLP(
                in_channels=q_channels,
                hidden_channels=int(q_channels * mlp_ratio),
                out_channels=q_channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    
    def forward(self, q_point: Point, kv_point: Point):
        # pdb.set_trace()
        q_shortcut = q_point.feat
        q_point = self.q_cpe(q_point)
        q_point.feat = q_shortcut + q_point.feat  # xCPE+x, (N,32)
        q_shortcut = q_point.feat

        kv_shortcut = kv_point.feat
        kv_point = self.kv_cpe(kv_point)
        kv_point.feat = kv_shortcut + kv_point.feat  # xCPE+x, (N,32)
        kv_shortcut = kv_point.feat

        if self.pre_norm:  # True
            q_point = self.q_norm1(q_point)  # Layer Norm, (N,32)
            kv_point = self.kv_norm1(kv_point)  # Layer Norm, (N,32)
        q_point = self.drop_path(self.attn(q_point, kv_point))  # Attention , 使用填补方式来执行attention，以至于batch并行

        # ---- feature fusion ----
        # q_point.feat = q_shortcut + q_point.feat  # Attention的残差连接
        if(self.tm_feat == "channel_scale" or self.tm_feat == "b_channel_scale"):
            feat_scale = torch.sigmoid(self.feat_scale) # 使用门限？？？
        else:
            feat_scale = self.feat_scale
        if(self.tm_feat == "b_channel_scale" or self.tm_feat == "b_lr_scale"):
            q_point.feat = (1-feat_scale) * q_shortcut + feat_scale * q_point.feat  # Attention的残差连接
        else:
            q_point.feat = q_shortcut + feat_scale * q_point.feat
        # ---- feature fusion ----

        if not self.pre_norm:  # False
            q_point = self.q_norm1(q_point)  # Layer Norm, (N,32)
            kv_point = self.kv_norm1(kv_point)  # Layer Norm, (N,32)
        
        q_shortcut = q_point.feat
        if self.pre_norm:  # True
            q_point = self.q_norm2(q_point)  # Layer Norm, (N,32)

        q_point = self.drop_path(self.mlp(q_point))  # FFN
        q_point.feat = q_shortcut + q_point.feat  # the residual connection of FFN
        if not self.pre_norm: # False
            q_point = self.q_norm2(q_point)  # Layer Norm, (N,32)

        q_point.sparse_conv_feat = q_point.sparse_conv_feat.replace_feature(q_point.feat)  # (N,32)

        return q_point

class TransferModule(PointModule):
    def __init__(
            self,
            q_channels=512,
            kv_channels=512,
            q_num_heads=32,
            kv_num_heads=32,
            q_patch_size=1024,
            kv_patch_size=1024,
            q_scale=None,
            kv_scale=None,
            mlp_ratio=4,
            qkv_bias=True,
            pre_norm=True,
            order_index=0,
            q_cpe_indice_key=f"stage0",
            kv_cpe_indice_key=f"stage0",
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            attn_drop=0.0,
            proj_drop=0.0,
            q_drop_path=0.0,
            kv_drop_path=0.0,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=True,
            upcast_softmax=True,

            tm_bidirectional=False,
            tm_feat=1.0,
            tm_restomer=False
    ):
        super().__init__()
        self.tm_bidirectional = tm_bidirectional

        self.qkv_seg_cross_block = CrossBlock(
            q_channels=q_channels,
            kv_channels=kv_channels,

            num_heads=q_num_heads,

            q_patch_size=q_patch_size,
            kv_patch_size=kv_patch_size,

            qkv_bias=qkv_bias,
            qk_scale=q_scale,

            q_cpe_indice_key=q_cpe_indice_key,
            kv_cpe_indice_key=kv_cpe_indice_key,

            mlp_ratio=mlp_ratio,
            pre_norm=pre_norm,
            norm_layer=norm_layer,
            act_layer=act_layer,

            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=q_drop_path,

            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,

            tm_feat=tm_feat,
            tm_restomer=tm_restomer

        )

    def forward(self, cond_point, seg_point):

        # c_point = self.qkv
        seg_point = self.qkv_seg_cross_block(seg_point, cond_point)

        return cond_point, seg_point


@MODELS.register_module("PT-v3m1-noise")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),

        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),

        diffusion_task=False,

        c_stride=(4, 4),
        c_enc_depths=(2, 2, 2),
        c_enc_channels=(32, 64, 128),
        c_enc_num_head=(2, 4, 8),
        c_enc_patch_size=(1024, 1024, 1024),
        c_dec_depths=(2, 2),
        c_dec_channels=(64, 64),
        c_dec_num_head=(4, 4),
        c_dec_patch_size=(1024, 1024),

        num_classes=20,
        T_dim=128,
        tm_bidirectional=False,
        tm_feat=1.0,
        tm_restomer=False,

        skip_connection_mode="add",
        b_factor=[1.0, 1.0, 1.0, 1.0],
        s_factor=[1.0, 1.0, 1.0, 1.0],
        skip_connection_scale=False,
        skip_connection_scale_i=False
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        ## init
        self.diffusion_task = diffusion_task
        self.T_dim = T_dim

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        ## point cloud stem network 
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )
        
        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")


        if self.diffusion_task:
            # ---- T Embeding ----- #
            if (self.T_dim != -1):
                self.fc_t1 = nn.Linear(T_dim, 4 * T_dim)
                self.fc_t2 = nn.Linear(4 * T_dim, T_dim)
                self.activation = swish
            # ---- T Embeding ----- #
            c_in = 3
            self.c_embedding = Embedding(
                in_channels=c_in,
                embed_channels=c_enc_channels[0],
                norm_layer=bn_layer,
                act_layer=act_layer,
            )

            self.c_num_stages = len(c_enc_depths)
            # ---- Diffusion Model Encoder ----- #
            _c_enc_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(c_enc_depths))
            ]
            self.c_enc = PointSequential()
            for s in range(self.c_num_stages):
                _c_enc_drop_path_ = _c_enc_drop_path[
                    sum(c_enc_depths[:s]): sum(c_enc_depths[: s + 1])
                ]
                _enc = PointSequential()
                if s > 0:
                    _enc.add(
                        SerializedPooling(
                            in_channels=c_enc_channels[s - 1],
                            out_channels=c_enc_channels[s],
                            stride=c_stride[s - 1],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                            T_dim=T_dim
                        ),
                        name="down",
                    )
                for i in range(c_enc_depths[s]):
                    _enc.add(
                        Block(
                            channels=c_enc_channels[s],
                            num_heads=c_enc_num_head[s],
                            patch_size=c_enc_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=_c_enc_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            T_dim=T_dim,
                        ),
                        name=f"block{i}",
                    )
                if len(_enc) != 0:
                    self.c_enc.add(module=_enc, name=f"enc{s}")

            ## condition decoder
            _c_dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(c_dec_depths))
            ]
            self.c_dec = PointSequential()
            c_dec_channels = list(c_dec_channels) + [c_enc_channels[-1]]
            for s in reversed(range(self.c_num_stages - 1)):
                _c_dec_drop_path_ = _c_dec_drop_path[
                                   sum(c_dec_depths[:s]): sum(c_dec_depths[: s + 1])
                                   ]
                _c_dec_drop_path_.reverse()
                _dec = PointSequential()
                _dec.add(
                    SerializedUnpooling(
                        in_channels=c_dec_channels[s + 1],
                        skip_channels=c_enc_channels[s],
                        out_channels=c_dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        skip_connection_mode="add" if (skip_connection_mode == "add") else "cat",
                        skip_connection_scale=skip_connection_scale
                    ),
                    name="up",
                )
                for i in range(c_dec_depths[s]):
                    _dec.add(
                        Block(
                            channels=c_dec_channels[s],
                            num_heads=c_dec_num_head[s],
                            patch_size=c_dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=_c_dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            T_dim=T_dim,
                        ),
                        name=f"block{i}",
                    )
                self.c_dec.add(module=_dec, name=f"dec{s}")
            ### ----------------- Transfer Model ----------------- ###
            self.condition_transfer = TransferModule(

                q_channels=dec_channels[-1],
                kv_channels=c_dec_channels[-1],

                q_num_heads=enc_num_head[-1], # n_dec_num_head[-1]
                kv_num_heads=c_enc_num_head[-1], # c_dec_num_head[-1]

                q_patch_size=enc_patch_size[-1], # n_dec_patch_size
                kv_patch_size=c_enc_patch_size[-1], # c_dec_patch_size

                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,

                q_scale=qk_scale,
                kv_scale=qk_scale,

                attn_drop=attn_drop,
                proj_drop=proj_drop,

                q_drop_path=enc_drop_path[2],
                kv_drop_path=enc_drop_path[2],

                norm_layer=ln_layer,
                act_layer=act_layer,
                pre_norm=pre_norm,
                order_index=0,

                q_cpe_indice_key=f"stage{2}",
                kv_cpe_indice_key=f"stage{2}",

                enable_rpe=enable_rpe,
                enable_flash=enable_flash,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,

                tm_bidirectional=tm_bidirectional,
                tm_feat=tm_feat,
                tm_restomer=tm_restomer
            )

            # ---- Diffusion Model Recon Head ----- #
            self.cond_head = (
                nn.Linear(c_dec_channels[0], c_in)
                if num_classes > 0
                else nn.Identity()
            )

            self.seg_head = (
                nn.Linear(dec_channels[0], num_classes)
                if num_classes > 0
                else nn.Identity()
            )

            print("size is ==>> ",len(self.c_enc))
            print("c_dec size is ==>> ",len(self.c_dec))

    def forward(self, cond_point=None, seg_point=None):
        pdb.set_trace()
        
        if self.training:
            if (self.diffusion_task):
                c_point = Point(cond_point)
                c_point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
                c_point.sparsify()

                s_point = Point(seg_point)
                s_point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
                s_point.sparsify()

                # 2. T embeding, [N, 128]
                if (self.T_dim != -1 and "t_emb" in c_point.keys()):
                    c_t_emb = c_point['t_emb']
                    c_t_emb = self.fc_t1(c_t_emb)
                    c_t_emb = self.activation(c_t_emb)
                    c_t_emb = self.fc_t2(c_t_emb)
                    c_t_emb = self.activation(c_t_emb)
                    c_point['t_emb'] = c_t_emb # mlp

                ## embed
                s_point = self.embedding(s_point)
                c_point = self.c_embedding(c_point)

                ## cond encoder
                c_point = self.c_enc[0](c_point) # 32
                c_point = self.c_enc[1](c_point) # 64
                c_point = self.c_enc[2](c_point) # 128

                ## seg encoder
                s_point = self.enc(s_point) # 512

                ## condition to guide denoise
                c_point, s_point = self.condition_transfer(c_point, s_point)

                ## cond decode
                c_point = self.c_dec[0](c_point) # 64
                c_point = self.c_dec[1](c_point) # 64

                ## seg decode
                s_point = self.dec(s_point)

                s_point.feat = self.seg_head(s_point.feat).contiguous()
                c_point.feat = self.cond_head(c_point.feat).contiguous()

                return c_point, s_point

                # if self.training:
                    # return c_point, s_point
                # else:
                #     s_point['seg_logits'] = s_point.feat
                #     return s_point
        else:
            s_point = Point(seg_point)
            s_point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
            s_point.sparsify()

            s_point = self.embedding(s_point)
            s_point = self.enc(s_point) # 512

            s_point = self.dec(s_point)
            s_point.feat = self.seg_head(s_point.feat).contiguous()

            return s_point

