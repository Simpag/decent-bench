"""
JAX backend for interoperability_2.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("jax")`` call.

JAX arrays are immutable, so :meth:`set_item` rebinds the wrapper's underlying value
rather than mutating it.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import time_ns
from typing import Any, cast

import jax
import jax.numpy as jnp

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability_2._abstracts._backend import _Backend
from decent_bench.utils.interoperability_2._backend_manager import register_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices, SupportedFrameworks


def _unwrap(array: Any) -> Any:  # noqa: ANN401
    """Return the underlying value of an :class:`Array`, or pass ``array`` through."""
    return array.value if isinstance(array, Array) else array


@register_backend(SupportedFrameworks.JAX)  # noqa: PLR0904
class JaxBackend(_Backend):
    """JAX implementation of :class:`_Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        super().__init__(device)
        self._native_device: jax.Device = self.device_to_native(device)
        self._key: jax.Array = jax.random.key(time_ns())

    # Array creation

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return Array(jnp.zeros(shape, device=self._native_device))

    def zeros_like(self, array: Array) -> Array:
        return Array(jnp.zeros_like(array.value))

    def ones(self, shape: tuple[int, ...]) -> Array:
        return Array(jnp.ones(shape, device=self._native_device))

    def ones_like(self, array: Array) -> Array:
        return Array(jnp.ones_like(array.value))

    def eye(self, n: int) -> Array:
        return Array(jnp.eye(n, device=self._native_device))

    def eye_like(self, array: Array) -> Array:
        v = array.value
        rows, cols = v.shape[-2:]
        return Array(jnp.eye(rows, cols, dtype=v.dtype, device=v.device))

    def device_to_native(self, device: SupportedDevices) -> jax.Device:
        if device == SupportedDevices.CPU:
            return jax.devices("cpu")[0]
        if device == SupportedDevices.GPU:
            return jax.devices("gpu")[0]
        raise ValueError(f"Unsupported device for JAX: {device}")

    def device_of(self, array: Array) -> SupportedDevices:
        platform = array.value.device.platform
        if platform == "gpu":
            return SupportedDevices.GPU
        if platform == "cpu":
            return SupportedDevices.CPU
        raise TypeError(f"Unsupported JAX platform: {platform}")

    # Array manipulation

    def copy(self, array: Array) -> Array:
        return Array(jnp.array(array.value, copy=True))

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(jnp.stack([a.value for a in arrays], axis=dim))

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return Array(jnp.reshape(array.value, shape))

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        return Array(jnp.transpose(array.value, axes=dim))

    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.value.shape)

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        return Array(jnp.squeeze(array.value, axis=dim))

    def unsqueeze(self, array: Array, dim: int) -> Array:
        return Array(jnp.expand_dims(array.value, axis=dim))

    def diag(self, array: Array) -> Array:
        return Array(jnp.diag(array.value))

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        return dtype(array.value.item())

    # Linalg

    def dot(self, array1: Array, array2: Array) -> Array:
        return Array(jnp.dot(array1.value, array2.value))

    def matmul(self, array1: Array, array2: Array) -> Array:
        return Array(array1.value @ array2.value)

    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return Array(jnp.linalg.norm(array.value, ord=p, axis=dim, keepdims=keepdims))

    # Math reductions

    def sum(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.sum(array.value, axis=dim, keepdims=keepdims))

    def mean(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.mean(array.value, axis=dim, keepdims=keepdims))

    def min(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.min(array.value, axis=dim, keepdims=keepdims))

    def max(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.max(array.value, axis=dim, keepdims=keepdims))

    # Math elementwise — JAX arrays are immutable; "in-place" ops rebind the wrapper.
    # Operands may be Array or scalar (operator dunders pass either); ``Array | float``
    # covers both because PEP 484's numeric tower implicitly admits ``int``.

    def add(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(jnp.add(_unwrap(array1), _unwrap(array2)))

    def iadd[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = jnp.add(array1.value, _unwrap(array2))
        return array1

    def sub(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(jnp.subtract(_unwrap(array1), _unwrap(array2)))

    def isub[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = jnp.subtract(array1.value, _unwrap(array2))
        return array1

    def mul(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(jnp.multiply(_unwrap(array1), _unwrap(array2)))

    def imul[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = jnp.multiply(array1.value, _unwrap(array2))
        return array1

    def div(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(jnp.divide(_unwrap(array1), _unwrap(array2)))

    def idiv[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value = jnp.divide(array1.value, _unwrap(array2))
        return array1

    def pow(self, array: Array, p: float) -> Array:
        return Array(jnp.power(array.value, p))

    def ipow[T: Array](self, array: T, p: float) -> T:
        array.value = jnp.power(array.value, p)
        return array

    def negative(self, array: Array) -> Array:
        return Array(jnp.negative(array.value))

    def absolute(self, array: Array) -> Array:
        return Array(jnp.abs(array.value))

    def sqrt(self, array: Array) -> Array:
        return Array(jnp.sqrt(array.value))

    # Operators

    def sign(self, array: Array) -> Array:
        return Array(jnp.sign(array.value))

    def maximum(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(jnp.maximum(_unwrap(array1), _unwrap(array2)))

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.argmax(array.value, axis=dim, keepdims=keepdims))

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return Array(jnp.argmin(array.value, axis=dim, keepdims=keepdims))

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        # JAX arrays are immutable; rebind the wrapper to a new array with `key` updated.
        array.value = array.value.at[key].set(value.value)

    def get_item(self, array: Array, key: ArrayKey) -> Array:
        return Array(array.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        self._key = jax.random.key(seed)

    def get_rng_state(self) -> dict[str, Any]:
        return {"jax_key": jax.random.key_data(self._key)}

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "jax_key" in state:
            self._key = jax.random.wrap_key_data(state["jax_key"])

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        sub = self._next_key()
        sample = jax.random.normal(sub, shape=shape).to_device(self._native_device)
        return Array(mean + std * sample)

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        sub = self._next_key()
        return Array(jax.random.uniform(sub, shape=shape, minval=low, maxval=high).to_device(self._native_device))

    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        v = array.value
        sub = self._next_key()
        sample = jax.random.normal(sub, shape=v.shape, dtype=v.dtype)
        return Array(mean + std * sample)

    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        v = array.value
        sub = self._next_key()
        return Array(jax.random.uniform(sub, shape=v.shape, dtype=v.dtype, minval=low, maxval=high))

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        v = array.value
        sub = self._next_key()
        indices = jax.random.choice(sub, a=v.shape[0], shape=(size,), replace=replace)
        return Array(v[indices])

    # internals

    def _next_key(self) -> jax.Array:
        """Split the stored key, advance state, return a sub-key for one draw."""
        self._key, sub = jax.random.split(self._key)
        return cast("jax.Array", sub)
