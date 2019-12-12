# -*- coding: utf-8 -*-
"""Functions to create activation layers.

"""
from .config import Config, ActivName 


def create_activ():
    """Creates an activation layer.

    Note:
        The parameters are configured in :attr:`Config.activ`.

    Returns:
        torch.nn.Module: The created activation layer.

    """
    kwargs = {k: v for k, v in Config.activ.items() if k not in ['name']}
    if Config.activ.name is ActivName.RELU:
        from torch.nn import ReLU
        return ReLU()
    elif Config.activ.name is ActivName.LEAKY_RELU:
        from torch.nn import LeakyReLU
        return LeakyReLU(**kwargs)
