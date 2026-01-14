from abc import ABC, abstractmethod

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array


class CompressionScheme(ABC):
    """Scheme defining how messages are compressed when sent over the network."""

    @abstractmethod
    def compress(self, msg: Array) -> Array:
        """Apply compression and return a new, compressed message."""


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress(self, msg: Array) -> Array:
        return msg


class Quantization(CompressionScheme):
    """Scheme that rounds each element in a message to *significant_digits*."""

    def __init__(self, n_significant_digits: int):
        self.n_significant_digits = n_significant_digits

    def compress(self, msg: Array) -> Array:
        res = np.vectorize(lambda x: float(f"%.{self.n_significant_digits - 1}e" % x))(iop.to_numpy(msg))
        return iop.to_array_like(res, msg)
