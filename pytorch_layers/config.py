# -*- coding: utf-8 -*-
"""Configurations and Enums"""

from enum import Enum
from config import Config as _Config


class ActivMode(str, Enum):
    """Enum of the activation names."""
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'


class NormMode(str, Enum):
    """Enum of the normalization names."""
    BATCH = 'batch'
    INSTANCE = 'instance'
    GROUP = 'group'
    NONE = 'none'


class Dim(int, Enum):
    """Enum of the operation dimensionality."""
    ONE = 1
    TWO = 2
    THREE = 3


class InterpMode(str, Enum):
    """Enum of the interpolate mode."""
    LINEAR = 'linear'
    NEAREST = 'nearest'


class PaddingMode(str, Enum):
    """Enum of the padding mode."""
    ZEROS = 'zeros'
    CIRCULAR = 'circular'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'


class Config(_Config):
    """Global configurations for layer creation.
    
    """
    dim = 3
    """int: Dimensionality of the operations."""

    activ_mode = 'relu'
    """str: Activation name."""

    activ_kwargs = dict()
    """dict: Activation settings."""

    norm_mode = 'instance'
    """str: Normalization name."""

    norm_kwargs = dict(affine=True)
    """dict: Normalization settings."""

    interp_mode = 'nearest'
    """str: Interpolation mode."""

    interp_kwargs = dict()
    """str: Interpolation settings."""

    dropout = 0.2
    """float: Dropout rate."""

    avg_pool = dict()
    """dict: Average pooling settings."""

    padding_mode = 'zeros'
    """str: Padding mode."""
