#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch
import torch.nn.functional as F
from copy import deepcopy

from pytorch_layers import Config, Dim, ActivName, NormName, InterpMode
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
        self.l1 = torch.nn.Conv3d(2, 2, 3, stride=2, padding=1, bias=False)
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
        self.l1 = torch.nn.Conv2d(2, 2, 1, bias=False)
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


class Model2(torch.nn.Module):
    """Model created using pytorch-layers."""
    def __init__(self):
        super().__init__()
        self.l1 = create_k1_conv(2, 2, bias=False)
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
        self.l1 = create_k3_conv(2, 2, stride=2, bias=False)
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

    str1 = RefModel3d().__str__()
    str2 = Model1().__str__().replace('Model1', 'RefModel3d')
    str2 = re.sub('[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    Config.dim = Dim.TWO
    Config.activ = dict(name=ActivName.LEAKY_RELU, negative_slope=0.02)
    Config.norm = dict(name=NormName.BATCH, track_running_stats=False)
    Config.interp['mode'] = InterpMode.LINEAR
    Config.dropout = 0.5
    Config.avg_pool = dict(stride=4)

    str1 = RefModel2d().__str__()
    str2 = Model2().__str__().replace('Model2', 'RefModel2d')
    str2 = re.sub('[ \t]+\(l6\).*\n', '', str2)
    assert str1 == str2

    x = torch.rand([1, 2, 8, 8])
    ref = RefModel2d().eval()
    t = Model2().eval()
    ref.load_state_dict(t.state_dict())
    y1 = ref(x)
    y2 = t(x)
    assert torch.all(torch.eq(y1, y2))

    attrs1 = {k: getattr(Config, k) for k in Config._get_attrs()}
    Config.save_json('config.json')
    Config.load_json('config.json')
    os.remove('config.json')
    attrs2 = {k: getattr(Config, k) for k in Config._get_attrs()}
    assert attrs1 == attrs2

    Config.show()


if __name__ == '__main__':
    test_layers()
