#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch
import torch.nn.functional as F
from copy import deepcopy

from pytorch_layers import Config
from pytorch_layers.config import Dim
from pytorch_layers.config import ActivMode, NormMode, InterpMode, PaddingMode
from pytorch_layers import create_conv, create_k1_conv, create_k3_conv
from pytorch_layers import create_activ
from pytorch_layers import create_dropout
from pytorch_layers import create_norm
from pytorch_layers import Interpolate, create_avg_pool, create_two_avg_pool
from pytorch_layers import create_adaptive_avg_pool, create_global_avg_pool
from pytorch_layers import create_interp, create_two_upsample


class RefModel3d(torch.nn.Module):
    """The 3D reference model."""
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv3d(2, 2, 1, bias=True)
        self.l2 = torch.nn.InstanceNorm3d(2, affine=True)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Dropout3d(0.2)
        self.l5 = torch.nn.AvgPool3d(2)
        self.l7 = torch.nn.AdaptiveAvgPool3d(1)
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='nearest', scale_factor=2)
        output = self.l7(output)
        return output


class RefModel2d(torch.nn.Module):
    """The 2D reference model."""
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(2, 2, 3, stride=2, bias=False, padding=1,
                                  padding_mode='reflect')
        self.l2 = torch.nn.BatchNorm2d(2, track_running_stats=False)
        self.l3 = torch.nn.LeakyReLU(0.02)
        self.l4 = torch.nn.Dropout2d(0.5)
        self.l5 = torch.nn.AvgPool2d(2, stride=4)
        self.l7 = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='bilinear', scale_factor=2)
        output = self.l7(output)
        return output


class RefModel2d2(torch.nn.Module):
    """The 2D reference model."""
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(2, 2, 3, padding=1, stride=2,
                                  padding_mode='circular', bias=False)
        self.l2 = torch.nn.Identity()
        self.l3 = torch.nn.LeakyReLU(0.02)
        self.l4 = torch.nn.Identity()
        self.l5 = torch.nn.AvgPool2d(2, stride=4)
        self.l7 = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='bilinear', scale_factor=2)
        output = self.l7(output)
        return output


class RefModel3d2(torch.nn.Module):
    """The 3D reference model."""
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv3d(2, 2, 3, padding=1, stride=2,
                                  padding_mode='replicate', bias=False)
        self.l2 = torch.nn.GroupNorm(2, 2)
        self.l3 = torch.nn.LeakyReLU(0.02)
        self.l4 = torch.nn.Identity()
        self.l5 = torch.nn.AvgPool3d(2, stride=4)
        self.l7 = torch.nn.AdaptiveAvgPool3d(1)
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='area', scale_factor=2)
        output = self.l7(output)
        return output


class RefModel1d(torch.nn.Module):
    """The 3D reference model."""
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv1d(2, 2, 1, bias=True)
        self.l2 = torch.nn.InstanceNorm1d(2, affine=True)
        self.l3 = torch.nn.ReLU()
        self.l4 = torch.nn.Dropout(0.2)
        self.l5 = torch.nn.AvgPool1d(2)
        self.l7 = torch.nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = F.interpolate(output, mode='linear', scale_factor=2)
        output = self.l7(output)
        return output


class Model2(torch.nn.Module):
    """Model created using pytorch-layers."""
    def __init__(self):
        super().__init__()
        self.l1 = create_k3_conv(2, 2, stride=2, bias=False)
        self.l2 = create_norm(2)
        self.l3 = create_activ()
        self.l4 = create_dropout()
        self.l5 = create_two_avg_pool()
        self.l6 = create_two_upsample()
        self.l7 = create_global_avg_pool()
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)
        output = self.l7(output)
        return output


class Model1(torch.nn.Module):
    """Model created using pytorch-layers."""
    def __init__(self):
        super().__init__()
        self.l1 = create_k1_conv(2, 2, bias=True, padding_mode='zeros')
        self.l2 = create_norm(2)
        self.l3 = create_activ()
        self.l4 = create_dropout()
        self.l5 = create_avg_pool(2)
        self.l6 = create_two_upsample()
        self.l7 = create_global_avg_pool()
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)
        output = self.l7(output)
        return output


def test_layers():

    config = Config()

    str1 = RefModel3d().__str__()
    str2 = Model1().__str__().replace('Model1', 'RefModel3d')
    str2 = re.sub(r'[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    config.dim = 1
    config.interp_mode = 'linear'
    str1 = RefModel1d().__str__()
    str2 = Model1().__str__().replace('Model1', 'RefModel1d')
    str2 = re.sub(r'[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    config.dim = 2
    config.activ_mode = 'leaky_relu'
    config.activ_kwargs = dict(negative_slope=0.02)
    config.norm_mode = 'batch'
    config.norm_kwargs = dict(track_running_stats=False)
    config.interp_mode = 'linear'
    config.dropout = 0.5
    config.padding_mode = 'reflect'
    config.avg_pool = dict(stride=4)

    str1 = RefModel2d().__str__()
    str2 = Model2().__str__().replace('Model2', 'RefModel2d')
    str2 = re.sub(r'[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    x = torch.rand([1, 2, 8, 8])
    ref = RefModel2d().eval()
    t = Model2().eval()
    ref.load_state_dict(t.state_dict())
    y1 = ref(x)
    y2 = t(x)
    assert torch.all(torch.eq(y1, y2))

    attrs1 = config.save_dict()
    config.save_json('config.json')
    config.load_json('config.json')
    os.remove('config.json')
    attrs2 = config.save_dict()
    assert attrs1 == attrs2

    config.dim = Dim.TWO
    config.activ_mode = ActivMode.LEAKY_RELU
    config.activ_kwargs = dict(negative_slope=0.02)
    config.norm_mode = NormMode.NONE
    config.norm_kwargs = dict(track_running_stats=False)
    config.interp_mode = InterpMode.LINEAR
    config.dropout = 0
    config.padding_mode = PaddingMode.CIRCULAR
    config.avg_pool = dict(stride=4)

    str1 = RefModel2d2().__str__()
    str2 = Model2().__str__().replace('Model2', 'RefModel2d2')
    str2 = re.sub(r'[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    config.dim = Dim.THREE
    config.activ_mode = ActivMode.LEAKY_RELU
    config.activ_kwargs = dict(negative_slope=0.02)
    config.norm_mode = NormMode.GROUP
    config.norm_kwargs = dict(num_groups=2)
    config.interp_mode = InterpMode.AREA
    config.dropout = 0
    config.padding_mode = PaddingMode.REPLICATE
    config.avg_pool = dict(stride=4)

    x = torch.rand([1, 2, 8, 8, 8])
    ref = RefModel3d2().eval()
    t = Model2().eval()
    print(t)
    ref.load_state_dict(t.state_dict())
    y1 = ref(x)
    y2 = t(x)
    assert torch.all(torch.eq(y1, y2))

    str1 = RefModel3d2().__str__()
    str2 = Model2().__str__().replace('Model2', 'RefModel3d2')
    str2 = re.sub(r'[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    print(config)


if __name__ == '__main__':
    test_layers()
