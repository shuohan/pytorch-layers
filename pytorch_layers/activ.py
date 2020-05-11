# -*- coding: utf-8 -*-
"""Functions to create activation layers.

"""
from .config import Config, ActivMode


def create_activ():
    """Creates an activation layer.

    Note:
        The type and parameters are configured in
        :attr:`pytorch_layers.Config.activ_mode` and
        :attr:`pytorch_layers.Config.activ_kwargs`.

    Returns:
        torch.nn.Module: The created activation layer.

    """
    config = Config()
    if ActivMode(config.activ_mode) is ActivMode.RELU:
        from torch.nn import ReLU
        return ReLU()
    elif ActivMode(config.activ_mode) is ActivMode.LEAKY_RELU:
        from torch.nn import LeakyReLU
        return LeakyReLU(**config.activ_kwargs)
