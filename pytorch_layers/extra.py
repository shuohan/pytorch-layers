# -*- coding: utf-8 -*-
"""Functions to create other layers.

"""
from .config import Config, Dim


def create_dropout():
    """Creates a spatial dropout layer.

    Note:
        The dropout probability is configured via
        :attr:`pytorch_layers.Config.dropout`, and the spatial dimension is
        configured by :attr:`pytorch_layers.Config.dim`.

    Returns:
        torch.nn.Module: The created dropout layer.

    """
    config = Config()
    if config.dropout == 0:
        from torch.nn import Identity
        return Identity()
    if config.dim is Dim.ONE:
        from torch.nn import Dropout
    elif config.dim is Dim.TWO:
        from torch.nn import Dropout2d as Dropout
    elif config.dim is Dim.THREE:
        from torch.nn import Dropout3d as Dropout
    return Dropout(config.dropout)
