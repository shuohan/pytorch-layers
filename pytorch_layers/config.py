# -*- coding: utf-8 -*-
"""Configurations and Enums"""

from enum import Enum
from config import Config as _Config


class ActivName(str, Enum):
    """Enum of the activation names."""
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'


class NormName(str, Enum):
    """Enum of the normalization names."""
    BATCH = 'batch'
    INSTANCE = 'instance'
    GROUP = 'group'


class Dim(int, Enum):
    """Enum of the operation dimensionality."""
    TWO = 2
    THREE = 3


class InterpMode(str, Enum):
    """Enum of the interpolate mode."""
    LINEAR = 'linear'
    NEAREST = 'nearest'


class Config(_Config):
    """Global configurations for layer creation.
    
    """
    dim = Dim.TWO
    """Dimensionality of the operations."""

    activ = {'name': ActivName.RELU}
    """Activation settings."""

    norm = {'name': NormName.INSTANCE, 'affine': True}
    """Normalization settings."""

    interp = {'mode': InterpMode.NEAREST}
    """Interpolation settings."""

    dropout = 0.2
    """Dropout rate."""

    avg_pool = {}
    """Average pooling."""

    @classmethod
    def load_dict(cls, config):
        """Loads configurations from a :class:`dict`.

        Args:
            config (dict): The configurations to load.

        Raises:
            KeyError: A field is not in the class attributes.

        """
        for key, value in config.items():
            if hasattr(cls, key):
                if key == 'activ':
                    value['name'] = ActivName(value['name'])
                elif key == 'norm':
                    value['name'] = NormName(value['name'])
                elif key == 'dim':
                    value = Dim(value)
                setattr(cls, key, value)
            else:
                raise KeyError('Config does not have the field %s' % key)
