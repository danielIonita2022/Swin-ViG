# 2023.03.16-Changed for building NexToU
#            Harbin Institute of Technology (Shenzhen), <pcshi@stu.hit.edu.cn>
from typing import Optional, List

# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch.nn import functional as F


##############################
#    Basic layers
##############################
def act_layer(act, inplace=True, neg_slope=1e-2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc, conv_op):
    # normalization layer
    norm = norm.lower()
    if norm == 'batch':
        if conv_op == nn.Conv2d:
            layer = nn.BatchNorm2d(nc, affine=True)
        elif conv_op == nn.Conv3d or conv_op == Conv3d_WS:
            layer = nn.BatchNorm3d(nc, affine=True)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)
    elif norm == 'group':
        if conv_op == nn.Conv2d:
            layer = nn.GroupNorm(nc // 4, nc, affine=True)
        elif conv_op == nn.Conv3d or conv_op == Conv3d_WS:
            layer = nn.GroupNorm(nc // 6, nc, affine=True)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)
    elif norm == 'instance':
        if conv_op == nn.Conv2d:
            layer = nn.InstanceNorm2d(nc, affine=True)
        elif conv_op == nn.Conv3d or conv_op == Conv3d_WS:
            layer = nn.InstanceNorm3d(nc, affine=True)
        else:
            raise NotImplementedError('conv operation [%s] is not found' % conv_op)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


def weight_standardization(weight: torch.Tensor, eps: float = 1e-5):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)


class Conv3d_WS(nn.Conv3d):
    """
    ## 3D Convolution Layer

    This extends the standard 3D Convolution layer and standardize the weights before the convolution step.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 eps: float = 1e-5):
        super(Conv3d_WS, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.conv3d(x, weight_standardization(self.weight, self.eps), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose3d_WS(nn.ConvTranspose3d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation=1,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 eps=1e-5
                 ):
        super(ConvTranspose3d_WS, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 3
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        return F.conv_transpose3d(
            x, weight_standardization(self.weight, self.eps), self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, conv_op=Conv3d_WS):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1], conv_op))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0., conv_op=Conv3d_WS, dropout_op=None):
        m = []
        self.conv_op = conv_op
        if self.conv_op == nn.Conv2d:
            self.batch_norm = nn.BatchNorm2d
            self.instance_norm = nn.InstanceNorm2d
            self.group_norm = nn.GroupNorm
            self.groups_num = 4
        elif self.conv_op == nn.Conv3d or self.conv_op == Conv3d_WS:
            self.batch_norm = nn.GroupNorm
            self.instance_norm = nn.InstanceNorm3d
            self.group_norm = nn.GroupNorm
            self.groups_num = 6  # modificat de la 6
        else:
            raise NotImplementedError('conv operation [%s] is not found' % self.conv_op)

        dropout_op = dropout_op
        dropout_op_kwargs = {}
        dropout_op_kwargs['p'] = drop
        for i in range(1, len(channels)):
            m.append(conv_op(channels[i - 1], channels[i], 1, bias=bias, groups=self.groups_num))

            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1], conv_op))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))

        super(BasicConv, self).__init__(*m)


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times k}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature
