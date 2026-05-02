"""
Module-level interoperability functions.

Each function delegates to the active backend resolved via
:func:`decent_bench.utils.interoperability_2.get_backend`. Calling any of these before
:func:`set_backend` raises :class:`RuntimeError`.

Keeping these as free functions (rather than methods on a delegating class) avoids the
boilerplate of a parallel pass-through hierarchy, while preserving precise type
signatures and IDE help.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability_2._backend_manager import get_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices

# Array creation


def zeros(shape: tuple[int, ...]) -> Array:
    """Create an array of zeros with the given shape."""
    return get_backend().zeros(shape)


def zeros_like(array: Array) -> Array:
    """Create an array of zeros matching the shape and type of ``array``."""
    return get_backend().zeros_like(array)


def ones(shape: tuple[int, ...]) -> Array:
    """Create an array of ones with the given shape."""
    return get_backend().ones(shape)


def ones_like(array: Array) -> Array:
    """Create an array of ones matching the shape and type of ``array``."""
    return get_backend().ones_like(array)


def eye(n: int) -> Array:
    """Create an ``n x n`` identity matrix."""
    return get_backend().eye(n)


def eye_like(array: Array) -> Array:
    """Create an identity matrix matching the trailing 2 dims of ``array``."""
    return get_backend().eye_like(array)


def device_to_native(device: SupportedDevices) -> Any:  # noqa: ANN401
    """Convert :class:`SupportedDevices` to the active backend's native device."""
    return get_backend().device_to_native(device)


def device_of(array: Array) -> SupportedDevices:
    """Return the :class:`SupportedDevices` of ``array``."""
    return get_backend().device_of(array)


# Array manipulation


def copy(array: Array) -> Array:
    """Return a copy of ``array``."""
    return get_backend().copy(array)


def stack(arrays: Sequence[Array], dim: int = 0) -> Array:
    """Stack a sequence of arrays along a new dimension."""
    return get_backend().stack(arrays, dim)


def reshape(array: Array, shape: tuple[int, ...]) -> Array:
    """Reshape ``array`` to ``shape``."""
    return get_backend().reshape(array, shape)


def transpose(array: Array, dim: tuple[int, ...] | None = None) -> Array:
    """Transpose ``array``; ``None`` reverses dimensions."""
    return get_backend().transpose(array, dim)


def shape(array: Array) -> tuple[int, ...]:
    """Return the shape of ``array``."""
    return get_backend().shape(array)


def squeeze(array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
    """Remove single-dimensional entries from ``array``."""
    return get_backend().squeeze(array, dim)


def unsqueeze(array: Array, dim: int) -> Array:
    """Insert a singleton dimension at ``dim``."""
    return get_backend().unsqueeze(array, dim)


def diag(array: Array) -> Array:
    """Diagonal: build from a vector or extract from a matrix."""
    return get_backend().diag(array)


def astype(array: Array, dtype: type[float | int | bool]) -> float | int | bool:
    """Cast a single-element ``array`` to a Python scalar of ``dtype``."""
    return get_backend().astype(array, dtype)


# Linalg


def dot(array1: Array, array2: Array) -> Array:
    """Dot product of two arrays."""
    return get_backend().dot(array1, array2)


def matmul(array1: Array, array2: Array) -> Array:
    """Matrix multiplication of two arrays."""
    return get_backend().matmul(array1, array2)


def norm(
    array: Array,
    p: float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Norm of ``array``."""
    return get_backend().norm(array, p, dim, keepdims)


# Math reductions


def sum(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Sum elements of ``array`` along ``dim``."""
    return get_backend().sum(array, dim, keepdims)


def mean(
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Mean of ``array`` along ``dim``."""
    return get_backend().mean(array, dim, keepdims)


def min(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Minimum of ``array`` along ``dim``."""
    return get_backend().min(array, dim, keepdims)


def max(  # noqa: A001
    array: Array,
    dim: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Maximum of ``array`` along ``dim``."""
    return get_backend().max(array, dim, keepdims)


# Math elementwise


def add(array1: Array, array2: Array) -> Array:
    """Element-wise addition."""
    return get_backend().add(array1, array2)


def iadd[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise addition."""
    return get_backend().iadd(array1, array2)


def sub(array1: Array, array2: Array) -> Array:
    """Element-wise subtraction."""
    return get_backend().sub(array1, array2)


def isub[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise subtraction."""
    return get_backend().isub(array1, array2)


def mul(array1: Array, array2: Array) -> Array:
    """Element-wise multiplication."""
    return get_backend().mul(array1, array2)


def imul[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise multiplication."""
    return get_backend().imul(array1, array2)


def div(array1: Array, array2: Array) -> Array:
    """Element-wise division."""
    return get_backend().div(array1, array2)


def idiv[T: Array](array1: T, array2: Array) -> T:
    """In-place element-wise division."""
    return get_backend().idiv(array1, array2)


def pow(array: Array, p: float) -> Array:  # noqa: A001
    """Raise ``array`` to power ``p``."""
    return get_backend().pow(array, p)


def ipow[T: Array](array: T, p: float) -> T:
    """In-place raise ``array`` to power ``p``."""
    return get_backend().ipow(array, p)


def negative(array: Array) -> Array:
    """Element-wise negation."""
    return get_backend().negative(array)


def absolute(array: Array) -> Array:
    """Element-wise absolute value."""
    return get_backend().absolute(array)


def sqrt(array: Array) -> Array:
    """Element-wise square root."""
    return get_backend().sqrt(array)


# Operators


def sign(array: Array) -> Array:
    """Element-wise sign."""
    return get_backend().sign(array)


def maximum(array1: Array, array2: Array) -> Array:
    """Element-wise maximum."""
    return get_backend().maximum(array1, array2)


def argmax(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """Index of maximum value along ``dim``."""
    return get_backend().argmax(array, dim, keepdims)


def argmin(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
    """Index of minimum value along ``dim``."""
    return get_backend().argmin(array, dim, keepdims)


def set_item(array: Array, key: ArrayKey, value: Array) -> None:
    """Set ``array[key] = value`` in place."""
    get_backend().set_item(array, key, value)


def get_item(array: Array, key: ArrayKey) -> Array:
    """Return ``array[key]``."""
    return get_backend().get_item(array, key)
