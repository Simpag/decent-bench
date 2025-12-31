import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.abstracts.scheme import CompressionScheme
from decent_bench.utils.array import Array


class NoCompression(CompressionScheme):
    """Scheme that leaves messages uncompressed."""

    def compress(self, msg: Array) -> Array:  # noqa: D102
        return msg


class Quantization(CompressionScheme):
    """Scheme that rounds each element in a message to *significant_digits*."""

    def __init__(self, n_significant_digits: int):
        self.n_significant_digits = n_significant_digits

    def compress(self, msg: Array) -> Array:  # noqa: D102
        res = np.vectorize(lambda x: float(f"%.{self.n_significant_digits - 1}e" % x))(iop.to_numpy(msg))
        return iop.to_array_like(res, msg)
