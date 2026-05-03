"""
NumPy backend for interoperability_2.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("numpy")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability._abstracts._backend import _Backend
from decent_bench.utils.interoperability._backend_manager import register_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices, SupportedFrameworks


def _unwrap(array: Any) -> Any:  # noqa: ANN401
    """
    Return the underlying value of an :class:`Array`, or pass ``array`` through.

    Typed as ``Any`` because operator dunders may pass either an :class:`Array` or a
    Python scalar; the strict abstract signature would force a ``cast`` at every call
    site without runtime benefit.
    """
    return array.value if isinstance(array, Array) else array


class NumpyBackend(_Backend):  # noqa: PLR0904
    """NumPy implementation of :class:`_Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        if device != SupportedDevices.CPU:
            raise ValueError(f"NumPy backend only supports CPU, got '{device.value}'.")
        super().__init__(device)
        self._rng: np.random.Generator = np.random.default_rng()

    # Array creation

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return Array(np.zeros(shape))

    def zeros_like(self, array: Array) -> Array:
        return Array(np.zeros_like(array.value))

    def ones(self, shape: tuple[int, ...]) -> Array:
        return Array(np.ones(shape))

    def ones_like(self, array: Array) -> Array:
        return Array(np.ones_like(array.value))

    def eye(self, n: int) -> Array:
        return Array(np.eye(n))

    def eye_like(self, array: Array) -> Array:
        v = array.value
        return Array(np.eye(*v.shape[-2:], dtype=v.dtype))

    def device_to_native(self, device: SupportedDevices) -> Any:  # noqa: ANN401
        # NumPy has no explicit device management; surface the request unchanged.
        return device

    def device_of(self, array: Array) -> SupportedDevices:  # noqa: ARG002
        return SupportedDevices.CPU

    # Array manipulation

    def copy(self, array: Array) -> Array:
        v = array.value
        if isinstance(v, np.ndarray | np.generic):
            return Array(np.copy(v))
        return Array(deepcopy(v))

    def to_numpy(self, array: Array) -> NDArray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = array.value
        if isinstance(v, np.ndarray):
            return v
        return np.asarray(v)

    def from_numpy(self, array: NDArray[Any]) -> Array:
        return Array(array)

    def to_array(self, array: float | bool) -> Array:
        return Array(np.array(array))

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(np.stack([a.value for a in arrays], axis=dim))

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return Array(np.reshape(array.value, shape))

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        return Array(np.transpose(array.value, axes=dim))

    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.value.shape)

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        return Array(np.squeeze(array.value, axis=dim))

    def unsqueeze(self, array: Array, dim: int) -> Array:
        return Array(np.expand_dims(array.value, axis=dim))

    def diag(self, array: Array) -> Array:
        return Array(np.diag(array.value))

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        v = array.value
        scalar = v.item() if hasattr(v, "item") else v
        return dtype(scalar)

    # Linalg

    def dot(self, array1: Array, array2: Array) -> Array:
        return Array(np.dot(array1.value, array2.value))

    def matmul(self, array1: Array, array2: Array) -> Array:
        return Array(array1.value @ array2.value)

    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return Array(np.linalg.norm(array.value, ord=p, axis=dim, keepdims=keepdims))

    # Math reductions

    def sum(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.sum(array.value, axis=dim, keepdims=keepdims))

    def mean(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.mean(array.value, axis=dim, keepdims=keepdims))

    def min(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.min(array.value, axis=dim, keepdims=keepdims))

    def max(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        return Array(np.max(array.value, axis=dim, keepdims=keepdims))

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(np.add(_unwrap(array1), _unwrap(array2)))

    def iadd[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value += _unwrap(array2)
        return array1

    def sub(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(np.subtract(_unwrap(array1), _unwrap(array2)))

    def isub[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value -= _unwrap(array2)
        return array1

    def mul(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(np.multiply(_unwrap(array1), _unwrap(array2)))

    def imul[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value *= _unwrap(array2)
        return array1

    def div(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(np.divide(_unwrap(array1), _unwrap(array2)))

    def idiv[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value /= _unwrap(array2)
        return array1

    def pow(self, array: Array, p: float) -> Array:
        return Array(np.power(array.value, p))

    def negative(self, array: Array) -> Array:
        return Array(np.negative(array.value))

    def absolute(self, array: Array) -> Array:
        return Array(np.abs(array.value))

    def sqrt(self, array: Array) -> Array:
        return Array(np.sqrt(array.value))

    # Operators

    def sign(self, array: Array) -> Array:
        return Array(np.sign(array.value))

    def maximum(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(np.maximum(_unwrap(array1), _unwrap(array2)))

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return Array(np.argmax(array.value, axis=dim, keepdims=keepdims))

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return Array(np.argmin(array.value, axis=dim, keepdims=keepdims))

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        array.value[key] = value.value

    def get_item(self, array: Array, key: ArrayKey) -> Array:
        return Array(array.value[key])

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
        return Array(self._rng.normal(loc=mean, scale=std, size=shape))

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(self._rng.uniform(low=low, high=high, size=shape))

    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        return Array(self._rng.normal(loc=mean, scale=std, size=array.value.shape))

    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        return Array(self._rng.uniform(low=low, high=high, size=array.value.shape))

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        return Array(self._rng.choice(array.value, size=size, replace=replace))


register_backend(SupportedFrameworks.NUMPY, NumpyBackend)
