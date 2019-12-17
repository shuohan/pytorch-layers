# -*- coding: utf-8 -*-

from torch.nn import Module
from torch.nn.functional import interpolate

from .config import Config, Dim, InterpMode


class Interpolate(Module):
    """Wrapper of :func:`torch.nn.functionals.interpolate`.

    """
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        output = interpolate(input, size=self.size,
                             scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)
        return output

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        info += ', align_corners=' + str(self.align_corners)
        return info


def create_avg_pool(kernel_size, **kwargs):
    """Creates an pooling layer.

    Note:
        The parameters are configured in :attr:`Config.avg_pool`. These
        parameters should be mutually exclusive from the input ``kwargs``.

    Returns:
        torch.nn.Module: The created average pooling layer.

    """
    if Config.dim is Dim.TWO:
        from torch.nn import AvgPool2d as AvgPool
    elif Config.dim is Dim.THREE:
        from torch.nn import AvgPool3d as AvgPool
    return AvgPool(kernel_size, **Config.avg_pool, **kwargs)


def create_two_avg_pool(**kwargs):
    """Creates pooling with kernel size 2."""
    return create_avg_pool(2, **kwargs)


def create_adaptive_avg_pool(output_size):
    """Creates adaptive average pooling.

    Args:
        output_size (int): The target output size.

    Returns:
        torch.nn.Module: The created adaptive average pooling layer.
    
    """
    if Config.dim is Dim.TWO:
        from torch.nn import AdaptiveAvgPool2d as AdaptiveAvgPool
    elif Config.dim is Dim.THREE:
        from torch.nn import AdaptiveAvgPool3d as AdaptiveAvgPool
    return AdaptiveAvgPool(output_size)


def create_global_avg_pool():
    """Creates global average pooling.
    
    Average the input image. The kernel size is equal to the image size. The
    output has spatial size 1.

    Returns:
        torch.nn.Module: The created pooling layer.

    """
    return create_adaptive_avg_pool(1)


def create_interp(size=None, scale_factor=None):
    """Creates an interpolate layer.

    See :func:`torch.nn.functionals.interpolate` for the inputs ``size`` and
    ``scale_factor``.

    Note:
        The type and other parameters of interpolate are configured in
        :attr:`Config.interpolate`.

    Returns:
        torch.nn.Module: The created interpolate layer.

    """
    if Config.interp['mode'] is InterpMode.LINEAR:
        if Config.dim is Dim.TWO:
            mode = 'bilinear'
        elif Config.dim is Dim.THREE:
            mode = 'trilinear'
    elif Config.interp['mode'] is InterpMode.NEAREST:
        mode = 'nearest'
        Config.interp['align_corners'] = None
    return Interpolate(size=size, scale_factor=scale_factor, mode=mode,
                       align_corners=Config.interp.get('align_corners'))


def create_two_upsample():
    """Creates interpolate with scale factor 2."""
    return create_interp(scale_factor=2)
