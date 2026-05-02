"""
NumPy backend for interoperability_2.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("numpy")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability_2._abstracts._backend import _Backend
from decent_bench.utils.interoperability_2._backend_manager import register_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices, SupportedFrameworks


def _wrap(value: NDArray[Any]) -> Array:
    """Wrap a raw NumPy value in an :class:`Array`. No-op if already wrapped."""
    if isinstance(value, Array):
        return value
    return Array(value)


def _unwrap(array: Array | NDArray[Any]) -> NDArray[Any]:
    """Return the underlying NumPy value of an :class:`Array` (or ``array`` itself)."""
    return cast("NDArray[Any]", array.value if isinstance(array, Array) else array)


@register_backend(SupportedFrameworks.NUMPY, aliases=("np",))  # noqa: PLR0904
class NumpyBackend(_Backend):
    """NumPy implementation of :class:`_Backend`."""

    def __init__(self) -> None:
        self._rng: np.random.Generator = np.random.default_rng()

    # Array creation

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return _wrap(np.zeros(shape))

    def zeros_like(self, array: Array) -> Array:
        return _wrap(np.zeros_like(_unwrap(array)))

    def ones(self, shape: tuple[int, ...]) -> Array:
        return _wrap(np.ones(shape))

    def ones_like(self, array: Array) -> Array:
        return _wrap(np.ones_like(_unwrap(array)))

    def eye(self, n: int) -> Array:
        return _wrap(np.eye(n))

    def eye_like(self, array: Array) -> Array:
        value: NDArray[Any] = _unwrap(array)
        return _wrap(np.eye(*value.shape[-2:], dtype=value.dtype))

    def device_to_native(self, device: SupportedDevices) -> Any:  # noqa: ANN401
        # NumPy has no explicit device management; surface the request unchanged.
        return device

    def device_of(self, array: Array) -> SupportedDevices:  # noqa: ARG002
        return SupportedDevices.CPU

    # Array manipulation

    def copy(self, array: Array) -> Array:
        value = _unwrap(array)
        if isinstance(value, np.ndarray | np.generic):
            return _wrap(np.copy(value))
        return _wrap(deepcopy(value))

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return _wrap(np.stack([_unwrap(a) for a in arrays], axis=dim))

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return _wrap(np.reshape(_unwrap(array), shape))

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        return _wrap(np.transpose(_unwrap(array), axes=dim))

    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(_unwrap(array).shape)

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        return _wrap(np.squeeze(_unwrap(array), axis=dim))

    def unsqueeze(self, array: Array, dim: int) -> Array:
        return _wrap(np.expand_dims(_unwrap(array), axis=dim))

    def diag(self, array: Array) -> Array:
        return _wrap(np.diag(_unwrap(array)))

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        value = _unwrap(array)
        scalar = value.item() if hasattr(value, "item") else value
        return dtype(scalar)

    # Linalg

    def dot(self, array1: Array, array2: Array) -> Array:
        return _wrap(np.dot(_unwrap(array1), _unwrap(array2)))

    def matmul(self, array1: Array, array2: Array) -> Array:
        return _wrap(_unwrap(array1) @ _unwrap(array2))

    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return _wrap(np.linalg.norm(_unwrap(array), ord=p, axis=dim, keepdims=keepdims))

    # Math reductions

    def sum(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return _wrap(np.sum(_unwrap(array), axis=dim, keepdims=keepdims))

    def mean(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return _wrap(np.mean(_unwrap(array), axis=dim, keepdims=keepdims))

    def min(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return _wrap(np.min(_unwrap(array), axis=dim, keepdims=keepdims))

    def max(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return _wrap(np.max(_unwrap(array), axis=dim, keepdims=keepdims))

    # Math elementwise

    def add(self, array1: Array, array2: Array) -> Array:
        return _wrap(np.add(_unwrap(array1), _unwrap(array2)))

    def iadd[T: Array](self, array1: T, array2: Array) -> T:
        value = _unwrap(array1)
        value += _unwrap(array2)
        return array1

    def sub(self, array1: Array, array2: Array) -> Array:
        return _wrap(np.subtract(_unwrap(array1), _unwrap(array2)))

    def isub[T: Array](self, array1: T, array2: Array) -> T:
        value = _unwrap(array1)
        value -= _unwrap(array2)
        return array1

    def mul(self, array1: Array, array2: Array) -> Array:
        return _wrap(np.multiply(_unwrap(array1), _unwrap(array2)))

    def imul[T: Array](self, array1: T, array2: Array) -> T:
        value = _unwrap(array1)
        value *= _unwrap(array2)
        return array1

    def div(self, array1: Array, array2: Array) -> Array:
        return _wrap(np.divide(_unwrap(array1), _unwrap(array2)))

    def idiv[T: Array](self, array1: T, array2: Array) -> T:
        value = _unwrap(array1)
        value /= _unwrap(array2)
        return array1

    def pow(self, array: Array, p: float) -> Array:
        return _wrap(np.power(_unwrap(array), p))

    def ipow[T: Array](self, array: T, p: float) -> T:
        value = _unwrap(array)
        value **= p
        return array

    def negative(self, array: Array) -> Array:
        return _wrap(np.negative(_unwrap(array)))

    def absolute(self, array: Array) -> Array:
        return _wrap(np.abs(_unwrap(array)))

    def sqrt(self, array: Array) -> Array:
        return _wrap(np.sqrt(_unwrap(array)))

    # Operators

    def sign(self, array: Array) -> Array:
        return _wrap(np.sign(_unwrap(array)))

    def maximum(self, array1: Array, array2: Array) -> Array:
        return _wrap(np.maximum(_unwrap(array1), _unwrap(array2)))

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return _wrap(np.argmax(_unwrap(array), axis=dim, keepdims=keepdims))

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return _wrap(np.argmin(_unwrap(array), axis=dim, keepdims=keepdims))

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        _unwrap(array)[key] = _unwrap(value)

    def get_item(self, array: Array, key: ArrayKey) -> Array:
        return _wrap(_unwrap(array)[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        # Seed both the legacy global state and our owned Generator. The legacy state is
        # important because some downstream libraries (sklearn, pandas) consult it.
        np.random.seed(seed)  # noqa: NPY002
        self._rng = np.random.default_rng(seed)

    def get_rng_state(self) -> dict[str, Any]:
        return {
            "numpy_bit_generator_state": deepcopy(self._rng.bit_generator.state),
            "numpy_legacy_state": np.random.get_state(),  # noqa: NPY002
        }

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "numpy_bit_generator_state" in state:
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = state["numpy_bit_generator_state"]
        if "numpy_legacy_state" in state:
            np.random.set_state(state["numpy_legacy_state"])  # noqa: NPY002

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return _wrap(self._rng.normal(loc=mean, scale=std, size=shape))

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return _wrap(self._rng.uniform(low=low, high=high, size=shape))

    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        value = _unwrap(array)
        return _wrap(self._rng.normal(loc=mean, scale=std, size=value.shape))

    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        value = _unwrap(array)
        return _wrap(self._rng.uniform(low=low, high=high, size=value.shape))

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        return _wrap(self._rng.choice(_unwrap(array), size=size, replace=replace))
