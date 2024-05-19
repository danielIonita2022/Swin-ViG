# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.torch_nn import act_layer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.NexToU_Encoder_Decoder import Grapher


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 13, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=[48,192,192], in_dim=1, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim // 2),
            act_layer(act),
            nn.Conv3d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
            act_layer(act),
            nn.Conv3d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act, upsample_scale=2):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2 * upsample_scale,
                                           stride=upsample_scale, padding=upsample_scale // 2)
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = act_layer(act)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        img_size = opt.img_size

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        # self.stem = Stem(out_dim=channels[0], act=act)
        D, H, W = img_size
        self.pos_embed = nn.Parameter(torch.zeros(2, 1, D, H, W))
        DHW = D * H * W

        self.encoder = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks) - 1):
            self.encoder.append(Downsample(channels[i], channels[i+1]))
            DHW = DHW // 4
            # for j in range(blocks[i]):
            if i >= 3:
                self.encoder.append(
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=DHW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        ))
                idx += 1

        up_channels = opt.channels[::-1][1:] + [opt.channels[0]]
        self.decoder = nn.ModuleList([])
        decoder_idx = idx - 1
        for i in range(len(up_channels) - 1):
            self.decoder.append(DecoderBlock(up_channels[i] * 2, up_channels[i+1], act=opt.act))
            DHW = DHW * 2

            #for j in range(up_channels[i]):
            self.decoder.append(
                Seq(Grapher(up_channels[i] * 2, num_knn[decoder_idx], min(decoder_idx // 4 + 1, max_dilation), conv, act, norm,
                            bias, stochastic, epsilon, reduce_ratios[i], n =DHW, drop_path=dpr[decoder_idx],
                            relative_pos=True),
                    FFN(up_channels[i] * 2, up_channels[i], act=act, drop_path=dpr[decoder_idx])))

            decoder_idx -= 1

        self.final_conv = nn.Conv3d(up_channels[-1] * 2, 13, kernel_size=1)
        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = inputs
        B, C, D, H, W = x.shape
        print("Input shape: ", B, C, D, H, W)
        skips = []
        for module in self.encoder:
            x = module(x)
            skips.append(x)
        skips = skips[::-1]
        print("Passed through encoder")

        for i, decoder_block in enumerate(self.decoder):
            skip_connection = skips[i]
            x = decoder_block(x, skip_connection)

        x = self.final_conv(x)
        return x

@register_model
def pvig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, img_size = [48,192,192], num_classes=13, drop_path_rate=0.0, **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
            self.channels = [48, 96, 240, 384]  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings
            self.img_size = img_size

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    #model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, img_size = [48,192,192], num_classes=13, drop_path_rate=0.0, **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640]  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings
            self.img_size = img_size

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    #model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_m_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, img_size = [48,192,192], num_classes=13, drop_path_rate=0.0, **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 16, 2]  # number of basic blocks in the backbone
            self.channels = [96, 192, 384, 768]  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings
            self.img_size = img_size

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    #model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, img_size = [48,192,192], num_classes=13, drop_path_rate=0.0, **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [1, 1, 1, 1, 1, 1]  # number of basic blocks in the backbone
            self.channels = [30, 60, 120, 240, 480, 960]  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings
            self.img_size = img_size

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    #model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model