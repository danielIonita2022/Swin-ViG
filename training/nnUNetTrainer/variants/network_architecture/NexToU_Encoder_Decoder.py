import mlflow
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, maybe_convert_scalar_to_list, \
    get_matching_pool_op

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.torch_nn import BasicConv, batched_index_select, \
    act_layer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.torch_edge import DenseDilatedKnnGraph
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.pos_embed import get_2d_relative_pos_embed, \
    get_3d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
from torch_geometric.nn.conv import GATConv, GATv2Conv


class OptInit:
    def __init__(self, drop_path_rate=0., pool_op_kernel_sizes_len=4):
        self.k = [4, 8, 16] + [32] * (pool_op_kernel_sizes_len - 3)
        self.conv = 'gat'
        self.act = 'leakyrelu'
        self.norm = 'instance'
        self.bias = True
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = True
        self.drop_path = drop_path_rate
        # number of basic blocks in the backbone
        self.blocks = [1] * (pool_op_kernel_sizes_len - 2) + [1, 1]
        # number of reduce ratios in the backbone
        self.reduce_ratios = [4, 2, 1, 1] + [1] * (pool_op_kernel_sizes_len - 4)


class NexToU_Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_size: List[int],
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        img_shape_list = []
        n_size_list = []
        conv_layer_d_num = 2
        pool_op_kernel_sizes = strides[1:]
        if conv_op == nn.Conv2d:
            h, w = patch_size[0], patch_size[1]
            img_shape_list.append((h, w))
            n_size_list.append(h * w)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                img_shape_list.append((h, w))
                n_size_list.append(h * w)

        elif conv_op == nn.Conv3d:
            h, w, d = patch_size[0], patch_size[1], patch_size[2]
            img_shape_list.append((h, w, d))
            n_size_list.append(h * w * d)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k, d_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                d //= d_k
                img_shape_list.append((h, w, d))
                n_size_list.append(h * w * d)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        img_min_shape = img_shape_list[-1]

        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        self.opt = opt
        self.opt.img_min_shape = img_min_shape
        self.conv_layer_d_num = conv_layer_d_num
        self.opt.n_size_list = n_size_list

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()

            if s < conv_layer_d_num:
                stage_modules.append(StackedConvBlocks(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    nonlin_first))
            else:
                stage_modules.append(nn.Sequential(
                    StackedConvBlocks(n_conv_per_stage[s] - 1, conv_op, input_channels, features_per_stage[s],
                                      kernel_sizes[s], conv_stride,
                                      conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                      nonlin_kwargs, nonlin_first),
                    Efficient_ViG_blocks(features_per_stage[s], img_shape_list[s], s - conv_layer_d_num,
                                         conv_layer_d_num, opt=self.opt, conv_op=conv_op,
                                         norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, is_decoder=False)))

            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        # print("Encoder: ")
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class NexToU_Decoder(nn.Module):
    def __init__(self,
                 encoder: NexToU_Encoder,
                 patch_size: List[int],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        img_shape_list = []
        n_size_list = []
        conv_layer_d_num = 2
        pool_op_kernel_sizes = strides[1:]
        if encoder.conv_op == nn.Conv2d:
            h, w = patch_size[0], patch_size[1]
            img_shape_list.append((h, w))
            n_size_list.append(h * w)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                img_shape_list.append((h, w))
                n_size_list.append(h * w)

        elif encoder.conv_op == nn.Conv3d:
            h, w, d = patch_size[0], patch_size[1], patch_size[2]
            img_shape_list.append((h, w, d))
            n_size_list.append(h * w * d)

            for i in range(len(pool_op_kernel_sizes)):
                h_k, w_k, d_k = pool_op_kernel_sizes[i]
                h //= h_k
                w //= w_k
                d //= d_k
                img_shape_list.append((h, w, d))
                n_size_list.append(h * w * d)
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" % str(encoder.conv_op))

        img_min_shape = img_shape_list[-1]

        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        self.opt = opt
        self.opt.img_min_shape = img_min_shape
        self.conv_layer_d_num = conv_layer_d_num
        self.opt.n_size_list = n_size_list

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))

            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            if s < (n_stages_encoder - conv_layer_d_num):
                stages.append(nn.Sequential(
                    StackedConvBlocks(n_conv_per_stage[s - 1] - 1, encoder.conv_op, 2 * input_features_skip,
                                      input_features_skip,
                                      encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op,
                                      encoder.norm_op_kwargs,
                                      encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin,
                                      encoder.nonlin_kwargs, nonlin_first),
                    Efficient_ViG_blocks(input_features_skip, img_shape_list[n_stages_encoder - (s + 1)],
                                         n_stages_encoder - conv_layer_d_num - (s + 1), conv_layer_d_num, opt=self.opt,
                                         conv_op=encoder.conv_op,
                                         norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                                         dropout_op=encoder.dropout_op, is_decoder=True)))

            else:
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                    encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        # print("Decoder: ")
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0,
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, norm_op_kwargs=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            conv_op(in_features, hidden_features, 1, stride=1, padding=0),
            norm_op(hidden_features, **norm_op_kwargs),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            conv_op(hidden_features, out_features, 1, stride=1, padding=0),
            norm_op(out_features, **norm_op_kwargs),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, conv_op=nn.Conv3d,
                 dropout_op=nn.Dropout3d):
        super(MRConv, self).__init__()
        self.conv_op = conv_op
        self.nn = BasicConv([in_channels * 2, out_channels], act=act, norm=norm, bias=bias, drop=0., conv_op=conv_op,
                            dropout_op=dropout_op)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)

        if self.conv_op == nn.Conv2d:
            pass
        elif self.conv_op == nn.Conv3d:
            x = torch.unsqueeze(x, dim=4)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        x = self.nn(x)

        return x


class GraphAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.LeakyReLU, norm=None, bias=True, conv_op=nn.Conv3d, dropout=nn.Dropout3d):
        super(GraphAttentionConv2d, self).__init__()
        self.gatconv = GATv2Conv(in_channels, out_channels, heads=2, concat=False, bias=bias, share_weights=True)
        self.act = nn.LeakyReLU()
        self.norm = nn.BatchNorm3d(out_channels)
        #self.dropout = nn.Dropout3d(p=0.3)
        self.conv_op = conv_op

    def forward(self, x, edge_index, y=None):
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
        if self.conv_op == nn.Conv3d:
            B, C, D, H, W = x.shape
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        x = x.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

        x = self.gatconv(x, edge_index)
        if self.conv_op == nn.Conv2d:
            x = x.view(B, -1, H, W)
        elif self.conv_op == nn.Conv3d:
            x = x.view(B, -1, D, H, W)
        x = self.norm(x)
        x = self.act(x)

        return x


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, conv_op=nn.Conv3d,
                 dropout_op=nn.Dropout3d):
        super(GraphConv, self).__init__()
        if conv == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias, conv_op, dropout_op)
        elif conv == 'gat':
            self.gconv = GraphAttentionConv2d(in_channels, out_channels, act, norm, bias, conv_op, dropout_op)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv(GraphConv):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, conv_op=nn.Conv3d, dropout_op=nn.Dropout3d):
        super(DyGraphConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, conv_op, dropout_op)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.conv = conv
        neighbors = 9
        mlflow.log_param("k", neighbors)
        self.dilated_knn_graph = DenseDilatedKnnGraph(neighbors, dilation, stochastic, epsilon)
        self.conv_op = conv_op
        self.dropout_op = nn.Dropout3d(0.2)
        if self.conv_op == nn.Conv2d:
            self.avg_pool = F.avg_pool2d
        elif self.conv_op == nn.Conv3d:
            self.avg_pool = F.avg_pool3d
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

    def forward(self, x, relative_pos=None):
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
        elif self.conv_op == nn.Conv3d:
            B, C, D, H, W = x.shape
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        y = None
        if self.r > 1:
            y = self.avg_pool(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        if self.conv == 'gat':
            x = x.view(B, C, D, H, W)
        x = super(DyGraphConv, self).forward(x, edge_index, y)
        if self.conv != 'gat':
            if self.conv_op == nn.Conv2d:
                return x.reshape(B, -1, H, W).contiguous()
            elif self.conv_op == nn.Conv3d:
                return x.reshape(B, -1, D, H, W).contiguous()
            else:
                raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
        x = self.dropout_op(x)
        return x


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False,
                 conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.conv_op = conv_op
        self.fc1 = nn.Sequential(
            conv_op(in_channels, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels),
        )
        self.graph_conv = DyGraphConv(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                      act, norm, bias, stochastic, epsilon, r, conv_op, dropout_op)
        self.fc2 = nn.Sequential(
            conv_op(in_channels * 2, in_channels, 1, stride=1, padding=0),
            norm_op(in_channels),
        )
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = DropPath(0.2)
        self.relative_pos = None
        if relative_pos:

            if self.conv_op == nn.Conv2d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                                                                                            int(n ** (
                                                                                                        1 / 2))))).unsqueeze(
                    0).unsqueeze(1)
                relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False)
            elif self.conv_op == nn.Conv3d:
                relative_pos_tensor = torch.from_numpy(np.float32(get_3d_relative_pos_embed(in_channels,
                                                                                            int(n ** (
                                                                                                        1 / 3))))).unsqueeze(
                    0).unsqueeze(1)
                relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n // (r * r * r)), mode='bicubic', align_corners=False)
            else:
                raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, size_tuple):
        if self.conv_op == nn.Conv2d:
            H, W = size_tuple
            if relative_pos is None or H * W == self.n:
                return relative_pos
            else:
                N = H * W
                N_reduced = N // (self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

        elif self.conv_op == nn.Conv3d:
            H, W, D = size_tuple
            if relative_pos is None or H * W * D == self.n:
                return relative_pos
            else:
                N = H * W * D
                N_reduced = N // (self.r * self.r * self.r)
                return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        if self.conv_op == nn.Conv2d:
            B, C, H, W = x.shape
            size_tuple = (H, W)
            relative_pos = self._get_relative_pos(self.relative_pos, size_tuple)
        elif self.conv_op == nn.Conv3d:
            B, C, H, W, D = x.shape
            size_tuple = (H, W, D)
            relative_pos = self._get_relative_pos(self.relative_pos, size_tuple)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class Efficient_ViG_blocks(nn.Module):
    def __init__(self, channels, img_shape, index, conv_layer_d_num, opt=None, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, is_decoder=False, **kwargs):
        super(Efficient_ViG_blocks, self).__init__()

        blocks = []
        k = opt.k
        conv = opt.conv
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        drop_path = opt.drop_path
        reduce_ratios = opt.reduce_ratios
        blocks_num_list = opt.blocks
        n_size_list = opt.n_size_list
        img_min_shape = opt.img_min_shape
        if is_decoder:
            conv = 'gat'
        else:
            conv = 'mr'

        self.n_blocks = sum(blocks_num_list)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        sum_blocks = sum(blocks_num_list[conv_layer_d_num - 2:index])
        idx_list = [(k + sum_blocks) for k in range(0, blocks_num_list[index])]

        if conv_op == nn.Conv2d:
            H, W = img_shape
            nr_of_vertices = H * W
            max_dilation = (nr_of_vertices) // max(k)
        elif conv_op == nn.Conv3d:
            H, W, D = img_shape
            nr_of_vertices = H * W * D
            max_dilation = (nr_of_vertices) // max(k)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)


        i = conv_layer_d_num - 2 + index
        for j in range(blocks_num_list[index]):
            idx = idx_list[j]

            blocks.append(nn.Sequential(
                Grapher(channels, k[i], min(idx // 4 + 1, max_dilation), 'mr', act, norm,
                            bias, stochastic, epsilon, reduce_ratios[i], n=nr_of_vertices, drop_path=dpr[idx],
                            relative_pos=True, conv_op=conv_op, norm_op=norm_op,
                            dropout_op=dropout_op),
                FFN(channels, channels * 4, act=act, drop_path=dpr[idx], conv_op=conv_op, norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs),
                Grapher(channels, k[i], min(idx // 4 + 1, max_dilation), 'gat', act, norm,
                            bias, stochastic, epsilon, reduce_ratios[i], n=nr_of_vertices, drop_path=dpr[idx],
                            relative_pos=True, conv_op=conv_op, norm_op=norm_op,
                            dropout_op=dropout_op),
                FFN(channels, channels * 4, act=act, drop_path=dpr[idx], conv_op=conv_op, norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs)))

        blocks = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        x = self.blocks(x)
        return x