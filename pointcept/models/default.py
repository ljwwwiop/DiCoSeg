import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

import numpy as np

from pointcept.models.losses import build_criteria, build_diff_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model
import math

import pdb

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorDiff(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits, diff_loss = self.backbone(input_dict)
        # pdb.set_trace()
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            loss = (diff_loss + loss)
            return dict(loss=loss,mse_loss=diff_loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits

        if 'diff_loss' in point.keys():
            loss = loss + point.diff_loss
            return_dict["loss"] = loss
            return_dict['diff_loss'] = point.diff_loss
            
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


## NOISE DIFF CDSEG


## weakly diffusion model
@MODELS.register_module()
class DefaultSegmentorV2DIFF(nn.Module):

    def __init__(
            self,
            num_classes=20,
            backbone=None,
            criteria=None,

            loss_type="EW",
            task_num=2,

            diffusion_task=False,
            T=1000,
            beta_start=0.0001,
            beta_end=0.02,
            noise_schedule="linear",
            T_dim=128,
            dm=False,
            dm_input="xt",
            dm_target="noise",
            dm_min_snr=None,
            c_in_channels=6,

        ):
        super().__init__()
        # self.seg_head = (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )

        self.backbone = build_model(backbone)
        self.criteria = build_diff_criteria(cfg=criteria,loss_type=loss_type,task_num=task_num)

        self.diffusion_task = diffusion_task
        self.num_classes = num_classes
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        self.T_dim = T_dim
        self.dm = dm
        self.dm_input = dm_input
        self.dm_target = dm_target
        self.dm_min_snr = dm_min_snr
        self.c_in_channels = c_in_channels

        ## diffusion params
        # self.T = 1000
        # self.T_dim = 128
        # self.beta_start = 0.001
        # self.beta_end = 0.005
        # self.noise_schedule = "linear"
        # self.dm_min_snr = None
        # self.dm_target = "noise"

        ## params
        self.eps = 1e-6
        self.Beta, self.Alpha ,self.Alpha_bar, self.Sigma, self.SNR = self.get_diffusion_hyperparams(
            noise_schedule=self.noise_schedule,
            T=self.T,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

        # ---- diffusion params ----

        self.Beta = self.Beta.float().cuda()
        self.Alpha = self.Alpha.float().cuda()
        self.Alpha_bar = self.Alpha_bar.float().cuda()
        self.Sigma = self.Sigma.float().cuda()
        self.SNR = self.SNR.float().cuda() if self.dm_min_snr is None else torch.clamp(self.SNR.float().cuda(),max=self.dm_min_snr)

    def add_gaussian_noise(self, pts, sigma=0.1, clamp=0.03):
        # input: (b, 3, n)

        assert (clamp > 0)
        # jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp)
        jittered_data = sigma * torch.randn_like(pts).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def add_random_noise(self, pts, sigma=0.1, clamp=0.03):
        # input: (b, 3, n)

        assert (clamp > 0)
        #         jittered_data = torch.clamp(sigma * torch.rand_like(pts), -1 * clamp, clamp).cuda()
        jittered_data = sigma * torch.rand_like(pts).cuda()
        jittered_data = jittered_data + pts

        return jittered_data


    def add_laplace_noise(self, pts, sigma=0.1, clamp=0.03, loc=0.0, scale=1.0):
        # input: (b, 3, n)

        assert (clamp > 0)
        laplace_distribution = torch.distributions.Laplace(loc=loc, scale=scale)
        jittered_data = sigma * laplace_distribution.sample(pts.shape).cuda()
        # jittered_data = torch.clamp(sigma * laplace_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def add_possion_noise(self, pts, sigma=0.1, clamp=0.03, rate=3.0):
        # input: (b, 3, n)

        assert (clamp > 0)
        poisson_distribution = torch.distributions.Poisson(rate)
        jittered_data = sigma * poisson_distribution.sample(pts.shape).cuda()
        # jittered_data = torch.clamp(sigma * poisson_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
        jittered_data = jittered_data + pts

        return jittered_data


    def get_diffusion_hyperparams(
            self,
            noise_schedule,
            beta_start,
            beta_end,
            T
        ):
        """
        Compute diffusion process hyperparameters

        Parameters:
        T (int):                    number of diffusion steps
        beta_0 and beta_T (float):  beta schedule start/end value,
                                    where any beta_t in the middle is linearly interpolated

        Returns:
        a dictionary of diffusion hyperparameters including:
            T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
            These cpu tensors are changed to cuda tensors on each individual gpu
        """

        # Beta = torch.linspace(noise_schedule,beta_start, beta_end, T)
        Beta = self.get_diffusion_betas(
            type=noise_schedule,
            start=beta_start,
            stop=beta_end,
            T=T
        )
        # at = 1 - bt
        Alpha = 1 - Beta
        # at_
        Alpha_bar = Alpha + 0
        # 方差
        Beta_tilde = Beta + 0
        for t in range(1, T):
            # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
            Alpha_bar[t] *= Alpha_bar[t - 1]
            # \tilde{\beta}_t = (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t) * \beta_t
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
        # 标准差
        Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
        Sigma[0] = 0.0

        '''
            SNR = at ** 2 / sigma ** 2
            at = sqrt(at_), sigma = sqrt(1 - at_)
            q(xt|x0) = sqrt(at_) * x0 + sqrt(1 - at_) * noise
        '''
        SNR = Alpha_bar / (1 - Alpha_bar)

        return Beta, Alpha, Alpha_bar, Sigma, SNR

    def get_diffusion_betas(self, type='linear', start=0.0001, stop=0.02, T=1000):
        """Get betas from the hyperparameters."""
        if type == 'linear':
            # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
            # To be used with Gaussian diffusion models in continuous and discrete
            # state spaces.
            # To be used with transition_mat_type = 'gaussian'
            scale = 1000 / T
            beta_start = scale * start
            beta_end = scale * stop
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        elif type == 'cosine':
            # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
            # To be used with transition_mat_type = 'uniform'.
            steps = T + 1
            s = 0.008
            # t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            t = torch.linspace(start, stop, steps, dtype=torch.float64) / T
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)


        elif type == 'sigmoid':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
            # To be used with absorbing state models.
            # ensures that the probability of decaying to the absorbing state
            # increases linearly over time, and is 1 for t = T-1 (the final time).
            # To be used with transition_mat_type = 'absorbing'
            start = -3
            end = 3
            tau = 1
            steps = T + 1
            t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            v_start = torch.tensor(start / tau).sigmoid()
            v_end = torch.tensor(end / tau).sigmoid()
            alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        elif type == "laplace":
            mu = 0.0
            b = 0.5
            lmb = lambda t: mu - b * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))

            snr_func = lambda t: torch.exp(lmb(t))
            alpha_func = lambda t: torch.sqrt(snr_func(t) / (1 + snr_func(t)))
            # sigma_func = lambda t: torch.sqrt(1 / (1 + snr_func(t)))

            timesteps = torch.linspace(0, 1, 1002)[1:-1]
            alphas_cumprod = []
            for t in timesteps:
                a = alpha_func(t) ** 2
                alphas_cumprod.append(a)
            alphas_cumprod = torch.cat(alphas_cumprod,dim=0)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError(type)

    def init_feature(self, input_dict):
        point = {}
        point["coord"] = input_dict["coord"]
        point["grid_coord"] = input_dict["grid_coord"]
        point["offset"] = input_dict["offset"]
        return point

    def calc_t_emb(self, ts, t_emb_dim):
        '''
            embed time steps into a higher dimension space
        '''
        assert t_emb_dim % 2 == 0

        if (ts.shape == 1):
            ts = ts.unsqueeze(1)
        half_dim = t_emb_dim // 2
        t_emb = np.log(10000) / (half_dim - 1)
        t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
        t_emb = t_emb.to(ts.device)

        t_emb = ts * t_emb
        t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)

        return t_emb

    def continuous_q_sample(self, x_0, t, noise=None):
        if (noise is None):
            noise = torch.normal(0, 1, size=x_0.shape, dtype=torch.float32)
        
        # xt = sqrt(at_) * x0 + sqrt(1-at_) * noise
        x_t = torch.sqrt(self.Alpha_bar[t]) * x_0 + ( torch.sqrt( 1 - self.Alpha_bar[t]) )* noise
        return x_t

    def inference(self, input_dict, eval=True, noise_level=None):

        if(noise_level is not None):
            input_dict["feat"] = self.add_gaussian_noise(input_dict["feat"],sigma=noise_level)
            #input_dict["feat"] = self.add_random_noise(input_dict["feat"],sigma=noise_level)
            #input_dict["feat"] = self.add_laplace_noise(input_dict["feat"],sigma=noise_level)

        if(self.diffusion_task):
            ### ---- PT V3 + DM ---- ###
            c_point = self.init_feature(input_dict)
            n_point = self.init_feature(input_dict)

            # ---- initial input ---- #
            n_point["feat"] = input_dict["feat"]

            if(self.c_in_channels == n_point["feat"].shape[-1]):
                c_point['feat'] = c_target = input_dict["feat"]
            else:
                c_point['feat'] = c_target = input_dict["coord"]

            t = 0
            if(self.dm and self.dm_input == "xt"):
                c_point['feat'] = torch.normal(0, 1, size=c_target.shape, dtype=torch.float32).cuda()
                t = self.T - 1
            # ---- initial input ---- #

            N = len(c_target)

            # ---- T steps ---- #
            ts = t * torch.ones((N, 1), dtype=torch.int64).cuda()
            if (self.T_dim != -1):
                c_point['t_emb'] = self.calc_t_emb(ts, t_emb_dim=self.T_dim).cuda()
            # ---- T steps ---- #

            # ---- pred c_epsilon and n_x0 ---- #
            c_point, n_point = self.backbone(c_point, n_point)
            # ---- pred c_epsilon and n_x0 ---- #

        if(eval):
            point = {}
            point['n_pred'] = n_point["feat"]
            point['n_target'] = input_dict['segment']
            point['loss_mode'] = "eval"
            loss = self.criteria(point)
            return dict(loss=loss, seg_logits=n_point["feat"])
        else:
            return dict(seg_logits=n_point["feat"])

    def forward(self, input_dict):
        pdb.set_trace()
        return_dict = dict()
        if self.training:
            if self.diffusion_task:
                point = {}
                # init data
                # two branch, seg branch, rec branch
                seg_point = self.init_feature(input_dict)
                cond_point = self.init_feature(input_dict)

                # init Point
                seg_point = Point(seg_point)
                cond_point = Point(cond_point)

                batch = seg_point['batch']
                B = len(torch.unique(batch))

                # init diffusion input
                seg_point['feat'] = input_dict['feat']
                
                # 4 or 3
                if (seg_point["feat"].shape[-1] == 4):
                    cond_point['feat'] = c_target = input_dict['feat']
                else:
                    cond_point['feat'] = c_target = input_dict['coord']


                # load diffusion step noise

                # time-sampling, time steps
                ts = torch.randint(0, self.T, size = (B, 1), dtype=torch.int64).cuda()
                # params ts 
                if self.T_dim != -1:
                    cond_point['t_emb'] = self.calc_t_emb(ts, t_emb_dim = self.T_dim)[batch, :].cuda()
                ts = ts[batch, :]

                # add noise
                cond_x0 = c_target
                cond_noise = torch.normal(0, 1, size=cond_x0.shape, dtype=torch.float32).cuda()
                cond_xt = self.continuous_q_sample(cond_x0, ts, cond_noise)# x0 + noise
                cond_point['feat'] = cond_xt 

                # ---- diffusion target ---- #
                if(self.dm_target == "noise"):
                    c_target = cond_noise # noise for MSE loss

                # ---- SNR Loss Weight ----
                if (self.dm_min_snr is not None):
                    point["snr_loss_weight"] = self.SNR[ts]

                # forward FFN
                cond_point, seg_point = self.backbone(cond_point, seg_point)

                point['c_pred'] = cond_point["feat"] # rec
                point['c_target'] = c_target # guassian noise

                point['n_pred'] = seg_point['feat'] # seg
                point['n_target'] = input_dict['segment']
                point['loss_mode'] = "train"
                # point['seg_logits'] = 
                loss = self.criteria(point)
                return dict(loss=loss)

        # test
        else:
            seg_point = self.init_feature(input_dict)

            # init Point
            seg_point = Point(seg_point)

            # init diffusion input
            seg_point['feat'] = input_dict['feat']
            
            seg_logits = self.backbone(seg_point=seg_point)
            return_dict["seg_logits"] = seg_logits['feat']
            
            return return_dict


