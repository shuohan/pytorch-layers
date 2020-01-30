# -*- coding: utf-8 -*-
"""Functions to create convolutional layers.

"""
import torch
import warnings
from .config import Config, Dim, PaddingMode


def create_conv(in_channels, out_channels, kernel_size, **kwargs):
    """Creates a convolutional layer.  
    Note:
        This function supports creating a 2D or 3D convolutional layer
        configured by :attr:`Config.dim`.

    Note:
        The function passes all keyword arguments directly to the Conv class.
        Check pytorch documentation for all keyword arguments (``bias``, for
        example).

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of kernel.

    Returns:
        torch.nn.Module: The created convolutional layer.

    """
    if 'padding_mode' in kwargs:
        message = ('"padding_mode" is ignored when creating conv. '
                   'Use Config to change it.')
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        kwargs.pop('padding_mode')

    if Dim(Config.dim) is Dim.TWO:
        from torch.nn import Conv2d as Conv
    elif Dim(Config.dim) is Dim.THREE:
        from torch.nn import Conv3d as Conv

    padding_mode = PaddingMode(Config.padding_mode)
    if padding_mode in [PaddingMode.REFLECT, PaddingMode.REPLICATE] \
            and 'padding' in kwargs:
        if Dim(Config.dim) is Dim.TWO:
            if padding_mode is PaddingMode.REFLECT:
                from torch.nn import ReflectionPad2d as Pad
            elif padding_mode is PaddingMode.REPLICATE:
                from torch.nn import ReplicationPad2d as Pad
        elif Dim(Config.dim) is Dim.THREE:
            if padding_mode is PaddingMode.REFLECT:
                raise NotImplementedError
            elif padding_mode is PaddingMode.REPLICATE:
                from torch.nn import ReplicationPad3d as Pad
        pad = Pad(kwargs['padding'])
        conv = Conv(in_channels, out_channels, kernel_size, **kwargs)
        model = torch.nn.Sequential(pad, conv)
    else:
        model = Conv(in_channels, out_channels, kernel_size,
                     padding_mode=Config.padding_mode, **kwargs)

    return  model


def create_k1_conv(in_channels, out_channels, **kwargs):
    """Creates a projection convolutional layer (kernel 1).

    Check :func:`create_conv` for more details.

    """
    return create_conv(in_channels, out_channels, 1, **kwargs)


def create_k3_conv(in_channels, out_channels, **kwargs):
    """Creates a convolutional layer with kernel 3 and "same" padding.

    Check :func:`create_conv` for more details.

    """
    return create_conv(in_channels, out_channels, 3, padding=1, **kwargs)
