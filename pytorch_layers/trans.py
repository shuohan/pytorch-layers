# -*- coding: utf-8 -*-

from torch.nn import Module
from torch.nn.functional import interpolate

from .config import Config, Dim, InterpMode


class Interpolate(Module):
    """Wrapper of :func:`torch.nn.functionals.interpolate`.

    """
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

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
        info += ', recompute_scale_factor=' + str(self.recompute_scale_factor)
        return info


def create_avg_pool(kernel_size, **kwargs):
    """Creates an pooling layer.

    Note:
        The parameters are configured in
        :attr:`pytorch_layers.Config.avg_pool_kwargs`. These parameters should
        be mutually exclusive from the input ``kwargs``.

    Returns:
        torch.nn.Module: The created average pooling layer.

    """
    config = Config()
    if config.dim is Dim.ONE:
        from torch.nn import AvgPool1d as AvgPool
    elif config.dim is Dim.TWO:
        from torch.nn import AvgPool2d as AvgPool
    elif config.dim is Dim.THREE:
        from torch.nn import AvgPool3d as AvgPool
    return AvgPool(kernel_size, **config.avg_pool_kwargs, **kwargs)


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
    config = Config()
    if config.dim is Dim.ONE:
        from torch.nn import AdaptiveAvgPool1d as AdaptiveAvgPool
    elif config.dim is Dim.TWO:
        from torch.nn import AdaptiveAvgPool2d as AdaptiveAvgPool
    elif config.dim is Dim.THREE:
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
        :meth:`pytorch_laayers.Config.interp_mode` and
        :attr:`pytorch_laayers.Config.interp_kwargs`.

    Returns:
        torch.nn.Module: The created interpolate layer.

    """
    config = Config()
    if config.interp_mode is InterpMode.LINEAR:
        if config.dim is Dim.ONE:
            mode = 'linear'
        elif config.dim is Dim.TWO:
            mode = 'bilinear'
        elif config.dim is Dim.THREE:
            mode = 'trilinear'
    elif config.interp_mode is InterpMode.NEAREST:
        mode = 'nearest'
        config.interp_kwargs['align_corners'] = None
    elif config.interp_mode is InterpMode.CUBIC:
        if config.dim is Dim.ONE:
            raise NotImplementedError
        elif config.dim is Dim.TWO:
            mode = 'bicubic'
        elif config.dim is Dim.THREE:
            raise NotImplementedError
    elif config.interp_mode is InterpMode.AREA:
        mode = 'area'
    return Interpolate(size=size, scale_factor=scale_factor, mode=mode,
                       **config.interp_kwargs)


def create_two_upsample():
    """Creates interpolate with scale factor 2."""
    return create_interp(scale_factor=2)
