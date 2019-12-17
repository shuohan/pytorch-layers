# -*- coding: utf-8 -*-
"""Functions to create normalization layers.

"""
from .config import Config, NormName, Dim


def create_norm(num_features):
    """Creates a normalization layer.

    Note:
        The normalization is configured via :attr:`Config.norm`, and the saptial
        dimension is configured via :attr:`Config.dim`.

    Args:
        num_features (int): The number of input channels.

    Returns:
        torch.nn.Module: The created normalization layer.

    """
    kwargs = {k: v for k, v in Config.norm.items()
              if k not in ['name', 'num_groups']}
    if Config.norm['name'] is NormName.GROUP:
        from torch.nn import GroupNorm
        return GroupNorm(Config.norm.num_groups, num_features, **kwargs)
    if Config.dim is Dim.TWO:
        if Config.norm['name'] is NormName.INSTANCE:
            from torch.nn import InstanceNorm2d
            return InstanceNorm2d(num_features, **kwargs)
        elif Config.norm['name'] == NormName.BATCH:
            from torch.nn import BatchNorm2d
            return BatchNorm2d(num_features, **kwargs)
    elif Config.dim == Dim.THREE:
        if Config.norm['name'] is NormName.INSTANCE:
            from torch.nn import InstanceNorm3d
            return InstanceNorm3d(num_features, **kwargs)
        elif Config.norm['name'] is NormName.BATCH:
            from torch.nn import BatchNorm3d
            return BatchNorm3d(num_features, **kwargs)
