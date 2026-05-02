from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from decent_bench.utils.array import Array


class _BackendArrayManipulation(ABC):
    @abstractmethod
    def copy(self, array: Array) -> Array:
        """
        Create a copy of the input array.

        Args:
            array (Array): Input array.

        Returns:
            Array: A copy of the input array.

        """

    @abstractmethod
    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        """
        Stack a sequence of arrays along a new dimension.

        Args:
            arrays (Sequence[Array]): Sequence of input arrays.
                or nested containers (list, tuple).
            dim (int): Dimension along which to stack the arrays.

        Returns:
            Array: Stacked array.

        """

    @abstractmethod
    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        """
        Reshape an array to the specified shape.

        Args:
            array (Array): Input array.
            shape (tuple[int, ...]): Desired shape for the output array.

        Returns:
            Array: Reshaped array in the same framework type as the input.

        """

    @abstractmethod
    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        """
        Transpose an array.

        Args:
            array (Array): Input array.
            dim (tuple[int, ...] | None): Desired dim order. If None, reverses the dimensions.

        Returns:
            Array: Transposed array.

        """

    @abstractmethod
    def shape(self, array: Array) -> tuple[int, ...]:
        """
        Get the shape of an array.

        Args:
            array (Array): Input array.

        Returns:
            tuple[int, ...]: Shape of the input array.

        """

    @abstractmethod
    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        """
        Remove single-dimensional entries from the shape of an array.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions to squeeze.
                If None, squeezes all single-dimensional entries.

        Returns:
            Array: Squeezed array.

        """

    @abstractmethod
    def unsqueeze(self, array: Array, dim: int) -> Array:
        """
        Insert a new dimension of size one at the specified position.

        Args:
            array (Array): Input array.
            dim (int): Position at which to insert the new dimension.

        Returns:
            Array: Unsqueezed array.

        """

    @abstractmethod
    def diag(self, array: Array) -> Array:
        """
        Create a diagonal matrix from a vector or extract a diagonal from a matrix.

        Args:
            array (Array): Input array.

        Returns:
            Array: Diagonal matrix or diagonal vector.

        """

    @abstractmethod
    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        """
        Cast a single-element array to a Python scalar of the specified type.

        Args:
            array (Array): The tensor.
            dtype (float | int | bool): The target data type.

        Returns:
            float | int | bool: The casted scalar value.

        Raises:
            TypeError: If the type is not supported.

        """
