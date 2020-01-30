# -*- coding: utf-8 -*-
"""Functions to create activation layers.

"""
from .config import Config, ActivMode


def create_activ():
    """Creates an activation layer.

    Note:
        The parameters are configured in :attr:`Config.activ`.

    Returns:
        torch.nn.Module: The created activation layer.

    """
    if ActivMode(Config.activ_mode) is ActivMode.RELU:
        from torch.nn import ReLU
        return ReLU()
    elif ActivMode(Config.activ_mode) is ActivMode.LEAKY_RELU:
        from torch.nn import LeakyReLU
        return LeakyReLU(**Config.activ_kwargs)
