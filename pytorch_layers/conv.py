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
        configured by :meth:`pytorch_layers.Config.dim`.

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
                   'Use pytorch_layers.Config to change it.')
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        kwargs.pop('padding_mode')
    
    config = Config()

    if config.dim is Dim.ONE:
        from torch.nn import Conv1d as Conv
    elif config.dim is Dim.TWO:
        from torch.nn import Conv2d as Conv
    elif config.dim is Dim.THREE:
        from torch.nn import Conv3d as Conv

    model = Conv(in_channels, out_channels, kernel_size,
                 padding_mode=config.padding_mode, **kwargs)

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
