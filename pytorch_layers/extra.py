# -*- coding: utf-8 -*-
"""Functions to create other layers.

"""
from .config import Config, Dim


def create_dropout():
    """Creates a spatial dropout layer.

    Note:
        The dropout probability is configured via :attr:`Config.dropout`, and
        the spatial dimension is configured by :attr:`Config.dim`.

    Returns:
        torch.nn.Module: The created dropout layer.

    """
    if Config.dropout == 0:
        from torch.nn import Identity
        return Identity()
    if Dim(Config.dim) is Dim.TWO:
        from torch.nn import Dropout2d as Dropout
    elif Dim(Config.dim) is Dim.THREE:
        from torch.nn import Dropout3d as Dropout
    return Dropout(Config.dropout)
