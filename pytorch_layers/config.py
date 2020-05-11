# -*- coding: utf-8 -*-
"""Configurations and Enums"""

from enum import Enum
from singleton_config import Config as _Config


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
    CUBIC = 'cubic'
    AREA = 'area'


class PaddingMode(str, Enum):
    """Enum of the padding mode."""
    ZEROS = 'zeros'
    CIRCULAR = 'circular'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'


class Config(_Config):
    """Global configurations for layer creation.

    Attributes:
        dim (int): Dimensionality of the operations.
        activ_mode (ActivMode): Activation mode.
        activ_kwargs (dict): Activation parameters.
        norm_mode (NormMode): Normalization mode.
        norm_kwargs (dict): Normalization parameters.
        interp_mode (InterpMode): Interpolation mode.
        interp_kwargs (dict): Interpolation parameters.
        padding_mode (PaddingMode): Padding mode.
        avg_pool (dict): The average pooling kwargs.

    """
    def __init__(self):
        super().__init__()
        self.add_config('dim', Dim.THREE, True)
        self.add_config('activ_mode', ActivMode.RELU, True)
        self.add_config('activ_kwargs', dict(), False)
        self.add_config('norm_mode', NormMode.INSTANCE, True)
        self.add_config('norm_kwargs', dict(affine=True), False)
        self.add_config('interp_mode', InterpMode.NEAREST, True)
        self.add_config('interp_kwargs', dict(), False)
        self.add_config('dropout', 0.2, False)
        self.add_config('padding_mode', PaddingMode.ZEROS, True)
        self.add_config('avg_pool', dict(), False)

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, d):
        if isinstance(d, Dim):
            self._dim = d
        elif isinstance(d, int):
            self._dim = Dim(d)
        else:
            assert False

    @property
    def activ_mode(self):
        return self._activ_mode

    @activ_mode.setter
    def activ_mode(self, m):
        if isinstance(m, ActivMode):
            self._activ_mode = m
        elif isinstance(m, str):
            self._activ_mode = ActivMode(m.lower())
        else:
            assert False

    @property
    def norm_mode(self):
        return self._norm_mode

    @norm_mode.setter
    def norm_mode(self, m):
        if isinstance(m, NormMode):
            self._norm_mode = m
        elif isinstance(m, str):
            self._norm_mode = NormMode(m.lower())
        else:
            assert False

    @property
    def interp_mode(self):
        return self._interp_mode

    @interp_mode.setter
    def interp_mode(self, m):
        if isinstance(m, InterpMode):
            self._interp_mode = m
        elif isinstance(m, str):
            self._interp_mode = InterpMode(m.lower())
        else:
            assert False

    @property
    def padding_mode(self):
        return self._padding_mode

    @padding_mode.setter
    def padding_mode(self, m):
        if isinstance(m, PaddingMode):
            self._padding_mode = m
        elif isinstance(m, str):
            self._padding_mode = PaddingMode(m.lower())
        else:
            assert False
