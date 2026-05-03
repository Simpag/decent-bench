"""
Module-level interoperability functions.

Each function delegates to the active backend cached in this module's ``_BACKEND``
slot. The slot is rebound by :func:`decent_bench.utils.interoperability_2.set_backend`
via :func:`_set_active_backend`. Calling any of these before ``set_backend`` raises
:class:`RuntimeError` via the sentinel's ``__getattr__``.

Caching the backend at module level (rather than calling ``get_backend()`` per
operation) drops the per-call cost from ~230 ns of dispatch overhead to ~40 ns at
small array sizes — matching the operator path's efficiency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from decent_bench.utils.array import Array
from decent_bench.utils.array._array import _NoBackendSet
from decent_bench.utils.types import ArrayKey, SupportedDevices

if TYPE_CHECKING:
    from decent_bench.utils.interoperability_2._abstracts._backend import _Backend


_BACKEND: _Backend = _NoBackendSet()  # type: ignore[assignment]


def _set_active_backend(backend: _Backend) -> None:
    """
    Bind the active backend.

    Called from :func:`decent_bench.utils.interoperability_2.set_backend`. Not part of
    the public API.
    """
    global _BACKEND  # noqa: PLW0603
    _BACKEND = backend

# Array creation


def zeros(shape: tuple[int, ...]) -> Array:
    """Create an array of zeros with the given shape."""
    return _BACKEND.zeros(shape)


def zeros_like(array: Array) -> Array:
    """Create an array of zeros matching the shape and type of ``array``."""
    return _BACKEND.zeros_like(array)


def ones(shape: tuple[int, ...]) -> Array:
    """Create an array of ones with the given shape."""
    return _BACKEND.ones(shape)


def ones_like(array: Array) -> Array:
    """Create an array of ones matching the shape and type of ``array``."""
    return _BACKEND.ones_like(array)


def eye(n: int) -> Array:
    """Create an ``n x n`` identity matrix."""
    return _BACKEND.eye(n)


def eye_like(array: Array) -> Array:
    """Create an identity matrix matching the trailing 2 dims of ``array``."""
    return _BACKEND.eye_like(array)


def device_to_native(device: SupportedDevices) -> Any:  # noqa: ANN401
    """Convert :class:`SupportedDevices` to the active backend's native device."""
    return _BACKEND.device_to_native(device)


def device_of(array: Array) -> SupportedDevices:
    """Return the :class:`SupportedDevices` of ``array``."""
    return _BACKEND.device_of(array)


# Array manipulation


def copy(array: Array) -> Array:
    """Return a copy of ``array``."""
    return _BACKEND.copy(array)


def stack(arrays: Sequence[Array], dim: int = 0) -> Array:
    """Stack a sequence of arrays along a new dimension."""
    return _BACKEND.stack(arrays, dim)


def reshape(array: Array, shape: tuple[int, ...]) -> Array:
    """Reshape ``array`` to ``shape``."""
    return _BACKEND.reshape(array, shape)


def transpose(array: Array, dim: tuple[int, ...] | None = None) -> Array:
    """Transpose ``array``; ``None`` reverses dimensions."""
    return _BACKEND.transpose(array, dim)


def shape(array: Array) -> tuple[int, ...]:
    """Return the shape of ``array``."""
    return _BACKEND.shape(array)


def squeeze(array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
    """Remove single-dimensional entries from ``array``."""
    return _BACKEND.squeeze(array, dim)


def unsqueeze(array: Array, dim: int) -> Array:
    """Insert a singleton dimension at ``dim``."""
    return _BACKEND.unsqueeze(array, dim)


def diag(array: Array) -> Array:
    """Diagonal: build from a vector or extract from a matrix."""
    return _BACKEND.diag(array)


def astype(array: Array, dtype: type[float | int | bool]) -> float | int | bool:
    """Cast a single-element ``array`` to a Python scalar of ``dtype``."""
    return _BACKEND.astype(array, dtype)


# Linalg


def dot(array1: Array, array2: Array) -> Array:
    """Dot product of two arrays."""
    return _BACKEND.dot(array1, array2)


def matmul(array1: Array, array2: Array) -> Array:
    """Matrix multiplication of two arrays."""
    return _BACKEND.matmul(array1, array2)


def norm(
    array: Array,
    p: float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Norm of ``array``."""
    return _BACKEND.norm(array, p, dim, keepdims)


# Math reductions


def sum(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Sum elements of ``array`` along ``dim``."""
    return _BACKEND.sum(array, dim, keepdims)


def mean(
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Mean of ``array`` along ``dim``."""
    return _BACKEND.mean(array, dim, keepdims)


def min(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Minimum of ``array`` along ``dim``."""
    return _BACKEND.min(array, dim, keepdims)


def max(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Maximum of ``array`` along ``dim``."""
    return _BACKEND.max(array, dim, keepdims)


# Math elementwise


def add(array1: Array, array2: Array) -> Array:
    """Element-wise addition."""
    return _BACKEND.add(array1, array2)


def iadd[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise addition."""
    return _BACKEND.iadd(array1, array2)


def sub(array1: Array, array2: Array) -> Array:
    """Element-wise subtraction."""
    return _BACKEND.sub(array1, array2)


def isub[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise subtraction."""
    return _BACKEND.isub(array1, array2)


def mul(array1: Array, array2: Array) -> Array:
    """Element-wise multiplication."""
    return _BACKEND.mul(array1, array2)


def imul[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise multiplication."""
    return _BACKEND.imul(array1, array2)


def div(array1: Array, array2: Array) -> Array:
    """Element-wise division."""
    return _BACKEND.div(array1, array2)


def idiv[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise division."""
    return _BACKEND.idiv(array1, array2)


def pow(array: Array, p: float) -> Array:  # noqa: A001
    """Raise ``array`` to power ``p``."""
    return _BACKEND.pow(array, p)


def ipow[T: Array](array: T, p: float) -> T:
    """In-place raise ``array`` to power ``p``."""
    return _BACKEND.ipow(array, p)


def negative(array: Array) -> Array:
    """Element-wise negation."""
    return _BACKEND.negative(array)


def absolute(array: Array) -> Array:
    """Element-wise absolute value."""
    return _BACKEND.absolute(array)


def sqrt(array: Array) -> Array:
    """Element-wise square root."""
    return _BACKEND.sqrt(array)


# Operators


def sign(array: Array) -> Array:
    """Element-wise sign."""
    return _BACKEND.sign(array)


def maximum(array1: Array, array2: Array) -> Array:
    """Element-wise maximum."""
    return _BACKEND.maximum(array1, array2)


def argmax(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """Index of maximum value along ``dim``."""
    return _BACKEND.argmax(array, dim, keepdims)


def argmin(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """Index of minimum value along ``dim``."""
    return _BACKEND.argmin(array, dim, keepdims)


def set_item(array: Array, key: ArrayKey, value: Array) -> None:
    """Set ``array[key] = value`` in place."""
    _BACKEND.set_item(array, key, value)


def get_item(array: Array, key: ArrayKey) -> Array:
    """Return ``array[key]``."""
    return _BACKEND.get_item(array, key)
