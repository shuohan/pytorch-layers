# -*- coding: utf-8 -*-
"""Functions to create normalization layers.

"""
from .config import Config, NormMode, Dim


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
    if NormMode(Config.norm_mode) is NormMode.GROUP:
        from torch.nn import GroupNorm
        num_groups = Config.norm_num_groups
        return GroupNorm(num_groups, num_features, **Config.norm_kwargs)
    elif NormMode(Config.norm_mode) is NormMode.NONE:
        from torch.nn import Identity
        return Identity()
    if Dim(Config.dim) is Dim.TWO:
        if NormMode(Config.norm_mode) is NormMode.INSTANCE:
            from torch.nn import InstanceNorm2d
            return InstanceNorm2d(num_features, **Config.norm_kwargs)
        elif NormMode(Config.norm_mode) is NormMode.BATCH:
            from torch.nn import BatchNorm2d
            return BatchNorm2d(num_features, **Config.norm_kwargs)
    elif Dim(Config.dim) is Dim.THREE:
        if NormMode(Config.norm_mode) is NormMode.INSTANCE:
            from torch.nn import InstanceNorm3d
            return InstanceNorm3d(num_features, **Config.norm_kwargs)
        elif NormMode(Config.norm_mode) is NormMode.BATCH:
            from torch.nn import BatchNorm3d
            return BatchNorm3d(num_features, **Config.norm_kwargs)
