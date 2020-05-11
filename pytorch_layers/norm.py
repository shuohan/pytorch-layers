# -*- coding: utf-8 -*-
"""Functions to create normalization layers.

"""
from .config import Config, NormMode, Dim


def create_norm(num_features):
    """Creates a normalization layer.

    Note:
        The normalization is configured via :meth:`pytorch_layers.Config.norm`,
        and :attr:`pytorch_layers.Config.norm_kwargs`, and the saptial dimension
        is configured via :attr:`pytorch_layers.Config.dim`.

    Args:
        num_features (int): The number of input channels.

    Returns:
        torch.nn.Module: The created normalization layer.

    """
    config = Config()
    if config.norm_mode is NormMode.GROUP:
        from torch.nn import GroupNorm
        kwargs = config.norm_kwargs.copy()
        num_groups = kwargs.pop('num_groups')
        return GroupNorm(num_groups, num_features, **kwargs)
    elif config.norm_mode is NormMode.NONE:
        from torch.nn import Identity
        return Identity()
    if config.dim is Dim.ONE:
        if config.norm_mode is NormMode.INSTANCE:
            from torch.nn import InstanceNorm1d
            return InstanceNorm1d(num_features, **config.norm_kwargs)
        elif config.norm_mode is NormMode.BATCH:
            from torch.nn import BatchNorm1d
            return BatchNorm1d(num_features, **config.norm_kwargs)
    elif config.dim is Dim.TWO:
        if config.norm_mode is NormMode.INSTANCE:
            from torch.nn import InstanceNorm2d
            return InstanceNorm2d(num_features, **config.norm_kwargs)
        elif config.norm_mode is NormMode.BATCH:
            from torch.nn import BatchNorm2d
            return BatchNorm2d(num_features, **config.norm_kwargs)
    elif config.dim is Dim.THREE:
        if config.norm_mode is NormMode.INSTANCE:
            from torch.nn import InstanceNorm3d
            return InstanceNorm3d(num_features, **config.norm_kwargs)
        elif config.norm_mode is NormMode.BATCH:
            from torch.nn import BatchNorm3d
            return BatchNorm3d(num_features, **config.norm_kwargs)
