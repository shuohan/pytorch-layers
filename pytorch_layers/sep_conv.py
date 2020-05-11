"""Functions to create separable convolutional layers.

"""
import torch
from .config import Config, Dim
from .norm import create_norm
from .activ import create_activ


class SeparableConv(torch.nn.Sequential):
    """Separable convolutional layer.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple[int]): The size of kernel.
        stride (int or tuple[int]): The number of strides.
        padding (int or tuple[int]): The padding size.
        dilation (int or tuple[int]): The dilation rate.

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        config = Config()
        if config.dim is Dim.ONE:
            from torch.nn import Conv1d as Conv
        elif config.dim is Dim.TWO:
            from torch.nn import Conv2d as Conv
        elif config.dim is Dim.THREE:
            from torch.nn import Conv3d as Conv

        dw_bias = not config.sep_conv_kwargs['norm_between']
        dw = Conv(self.in_channels, self.in_channels, self.kernel_size,
                  padding=self.padding, padding_mode=config.padding_mode,
                  groups=in_channels, bias=dw_bias, dilation=self.dilation,
                  stride=self.stride)
        pw = Conv(self.in_channels, self.out_channels, 1, bias=bias)

        self.add_module('depthwise', dw)
        if config.sep_conv_kwargs['norm_between']:
            self.add_module('norm', create_norm(in_channels))
        if config.sep_conv_kwargs['activ_between']:
            self.add_module('activ', create_activ())
        self.add_module('pointwise', pw)


def create_sep_conv(in_channels, out_channels, kernel_size, **kwargs):
    """Creates a separable convolutional layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of kernel.
        kwargs (dict): Other paramters.

    Returns:
        torch.nn.Module: The created separable convolutional layer.

    """
    return SeparableConv(in_channels, out_channels, kernel_size, **kwargs)
