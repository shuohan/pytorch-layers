#!/usr/bin/env python

import re
import torch
from collections import OrderedDict
from pytorch_layers import Config, create_sep_conv


def test_sep_conv():
    config = Config()
    conv1 = create_sep_conv(4, 16, 3, stride=2, padding=1, dilation=2,
                            bias=False)

    l1 = torch.nn.Conv3d(4, 4, 3, stride=2, padding=1, padding_mode='zeros',
                         dilation=2, groups=4)
    l2 = torch.nn.Conv3d(4, 16, 1, bias=False)
    conv2 = torch.nn.Sequential(OrderedDict([('depthwise', l1),
                                             ('pointwise', l2)]))
    str1 = conv1.__str__()
    str2 = conv2.__str__()
    str1 = re.sub(r'SeparableConv\(\n', r'Sequential(\n', str1)
    assert str1 == str2

    config.dim = 2
    config.sep_conv_kwargs['norm_between'] = True
    config.sep_conv_kwargs['activ_between'] = True
    config.activ_kwargs['affine'] = True
    conv1 = create_sep_conv(4, 16, 3)

    l1 = torch.nn.Conv2d(4, 4, 3, padding_mode='zeros', groups=4, bias=False)
    norm = torch.nn.InstanceNorm2d(4, affine=True)
    activ = torch.nn.ReLU()
    l2 = torch.nn.Conv2d(4, 16, 1)
    conv2 = torch.nn.Sequential(OrderedDict([('depthwise', l1),
                                             ('norm', norm),
                                             ('activ', activ),
                                             ('pointwise', l2)]))
    str1 = conv1.__str__()
    str2 = conv2.__str__()
    str1 = re.sub(r'SeparableConv\(\n', r'Sequential(\n', str1)
    assert str1 == str2

    print('all successful')


if __name__ == '__main__':
    test_sep_conv()
