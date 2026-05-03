"""
TensorFlow backend for interoperability_2.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("tensorflow")`` call.

TF eager Tensors are immutable, so :meth:`set_item` round-trips through numpy and the
in-place math operations rebind the wrapper's underlying value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import tensorflow as tf

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability_2._abstracts._backend import _Backend
from decent_bench.utils.interoperability_2._backend_manager import register_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices, SupportedFrameworks


def _unwrap(array: Any) -> Any:  # noqa: ANN401
    """Return the underlying value of an :class:`Array`, or pass ``array`` through."""
    return array.value if isinstance(array, Array) else array


@register_backend(SupportedFrameworks.TENSORFLOW, aliases=("tf",))  # noqa: PLR0904
class TensorflowBackend(_Backend):
    """TensorFlow implementation of :class:`_Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        super().__init__(device)
        self._native_device: str = self.device_to_native(device)
        self._generator: tf.random.Generator = tf.random.Generator.from_non_deterministic_state(alg="philox")

    # Array creation

    def zeros(self, shape: tuple[int, ...]) -> Array:
        with tf.device(self._native_device):
            return Array(tf.zeros(shape))

    def zeros_like(self, array: Array) -> Array:
        return Array(tf.zeros_like(array.value))

    def ones(self, shape: tuple[int, ...]) -> Array:
        with tf.device(self._native_device):
            return Array(tf.ones(shape))

    def ones_like(self, array: Array) -> Array:
        return Array(tf.ones_like(array.value))

    def eye(self, n: int) -> Array:
        with tf.device(self._native_device):
            return Array(tf.eye(n))

    def eye_like(self, array: Array) -> Array:
        v = array.value
        rows, cols = v.shape[-2:]
        return Array(tf.eye(rows, cols, dtype=v.dtype))

    def device_to_native(self, device: SupportedDevices) -> str:
        if device in (SupportedDevices.CPU, SupportedDevices.GPU):
            return f"/{device.value}:0"
        raise ValueError(f"Unsupported device for TensorFlow: {device}")

    def device_of(self, array: Array) -> SupportedDevices:
        device_str = array.value.device.lower()
        if "gpu" in device_str or "cuda" in device_str:
            return SupportedDevices.GPU
        return SupportedDevices.CPU

    # Array manipulation

    def copy(self, array: Array) -> Array:
        return Array(tf.identity(array.value))

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(tf.stack([a.value for a in arrays], axis=dim))

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return Array(tf.reshape(array.value, shape))

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        return Array(tf.transpose(array.value, perm=dim))

    def shape(self, array: Array) -> tuple[int, ...]:
        return cast("tuple[int, ...]", tuple(array.value.shape))

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        return Array(tf.squeeze(array.value, axis=dim))

    def unsqueeze(self, array: Array, dim: int) -> Array:
        return Array(tf.expand_dims(array.value, axis=dim))

    def diag(self, array: Array) -> Array:
        v = array.value
        rank = v.shape.ndims
        if rank == 1:
            return Array(tf.linalg.diag(v))
        if rank == 2:
            return Array(tf.linalg.diag_part(v))
        raise ValueError(f"diag requires a 1- or 2-D tensor, got rank {rank}")

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        return dtype(array.value.numpy().item())

    # Linalg

    def dot(self, array1: Array, array2: Array) -> Array:
        return Array(tf.tensordot(array1.value, array2.value, axes=1))

    def matmul(self, array1: Array, array2: Array) -> Array:
        # tf.matmul requires both operands to have ndim >= 2; fall back to tensordot
        # for the vector cases so semantics match numpy / torch / jax matmul.
        a, b = array1.value, array2.value
        if a.shape.ndims is None or b.shape.ndims is None or a.shape.ndims < 2 or b.shape.ndims < 2:
            return Array(tf.tensordot(a, b, axes=1))
        return Array(a @ b)

    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        v = array.value
        # tf.norm defaults differ from np.linalg.norm on 2-D inputs (operator vs.
        # Frobenius); match numpy's flat default by reducing over both trailing axes.
        axis = dim if dim is not None else (-2, -1) if v.ndim == 2 else None
        return Array(tf.norm(v, ord=p, axis=axis, keepdims=keepdims))

    # Math reductions

    def sum(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_sum(array.value, axis=dim, keepdims=keepdims))

    def mean(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_mean(array.value, axis=dim, keepdims=keepdims))

    def min(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_min(array.value, axis=dim, keepdims=keepdims))

    def max(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(tf.reduce_max(array.value, axis=dim, keepdims=keepdims))

    # Math elementwise — TF Tensors are immutable; "in-place" ops rebind the wrapper.
    # Operands may be Array or scalar (operator dunders pass either); ``Array | float``
    # covers both because PEP 484's numeric tower implicitly admits ``int``.

    def add(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(tf.add(_unwrap(array1), _unwrap(array2)))

    def iadd[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = tf.add(array1.value, _unwrap(array2))
        return array1

    def sub(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(tf.subtract(_unwrap(array1), _unwrap(array2)))

    def isub[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = tf.subtract(array1.value, _unwrap(array2))
        return array1

    def mul(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(tf.multiply(_unwrap(array1), _unwrap(array2)))

    def imul[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = tf.multiply(array1.value, _unwrap(array2))
        return array1

    def div(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(tf.divide(_unwrap(array1), _unwrap(array2)))

    def idiv[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = tf.divide(array1.value, _unwrap(array2))
        return array1

    def pow(self, array: Array, p: float) -> Array:
        return Array(tf.pow(array.value, p))

    def ipow[T: Array](self, array: T, p: float) -> T:
        array.value = tf.pow(array.value, p)
        return array

    def negative(self, array: Array) -> Array:
        return Array(tf.negative(array.value))

    def absolute(self, array: Array) -> Array:
        return Array(tf.abs(array.value))

    def sqrt(self, array: Array) -> Array:
        return Array(tf.sqrt(array.value))

    # Operators

    def sign(self, array: Array) -> Array:
        return Array(tf.sign(array.value))

    def maximum(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(tf.maximum(_unwrap(array1), _unwrap(array2)))

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        v = array.value
        if dim is None:
            flat = tf.argmax(tf.reshape(v, [-1]), axis=0)
            if keepdims:
                ndim = v.shape.ndims or 0
                return Array(tf.reshape(flat, [1] * ndim))
            return Array(flat)
        out = tf.argmax(v, axis=dim)
        if keepdims:
            out = tf.expand_dims(out, axis=dim)
        return Array(out)

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        v = array.value
        if dim is None:
            flat = tf.argmin(tf.reshape(v, [-1]), axis=0)
            if keepdims:
                ndim = v.shape.ndims or 0
                return Array(tf.reshape(flat, [1] * ndim))
            return Array(flat)
        out = tf.argmin(v, axis=dim)
        if keepdims:
            out = tf.expand_dims(out, axis=dim)
        return Array(out)

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        # TF eager tensors are immutable; round-trip through numpy so arbitrary indexing
        # patterns (slices, fancy indexing) Just Work. The wrapper is rebound to a fresh
        # tensor on the configured device. This is correct but allocates — algorithms
        # that hammer set_item in tight loops should consider numpy or pytorch.
        original = array.value
        np_array = original.numpy().copy()
        np_array[key] = np.asarray(value.value)
        with tf.device(self._native_device):
            array.value = tf.convert_to_tensor(np_array, dtype=original.dtype)

    def get_item(self, array: Array, key: ArrayKey) -> Array:
        return Array(array.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        tf.random.set_seed(seed)
        self._generator = tf.random.Generator.from_seed(seed, alg="philox")

    def get_rng_state(self) -> dict[str, Any]:
        return {"tf_generator_state": self._generator.state.numpy()}

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "tf_generator_state" in state:
            self._generator = tf.random.Generator.from_state(state["tf_generator_state"], alg="philox")

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        with tf.device(self._native_device):
            return Array(self._generator.normal(shape=shape, mean=mean, stddev=std))

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        with tf.device(self._native_device):
            return Array(self._generator.uniform(shape=shape, minval=low, maxval=high))

    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        v = array.value
        return Array(self._generator.normal(shape=tf.shape(v), mean=mean, stddev=std, dtype=v.dtype))

    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        v = array.value
        return Array(self._generator.uniform(shape=tf.shape(v), minval=low, maxval=high, dtype=v.dtype))

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        v = array.value
        n = v.shape[0]
        if replace:
            indices = self._generator.uniform(shape=(size,), minval=0, maxval=n, dtype=tf.int32)
        else:
            scores = self._generator.uniform(shape=(n,), dtype=tf.float32)
            indices = tf.cast(tf.math.top_k(scores, k=size).indices, tf.int32)
        return Array(tf.gather(v, indices))
