from abc import ABC, abstractmethod

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array


class NoiseScheme(ABC):
    """Scheme defining how noise impacts messages."""

    @abstractmethod
    def make_noise(self, msg: Array) -> Array:
        """Apply noise scheme without mutating the *msg* passed in."""


class NoNoise(NoiseScheme):
    """Scheme that leaves messages untouched."""

    def make_noise(self, msg: Array) -> Array:
        return msg


class GaussianNoise(NoiseScheme):
    """Scheme that applies Gaussian noise - that is, noise following a normal distribution."""

    def __init__(self, mean: float, sd: float):
        if sd < 0:
            raise ValueError("Standard deviation (sd) must be non-negative for Gaussian noise.")
        self.mean = mean
        self.sd = sd

    def make_noise(self, msg: Array) -> Array:
        return msg + iop.randn_like(msg, mean=self.mean, std=self.sd)
