from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array
from decent_bench.utils.types import ArrayKey


class _BackendOperators(ABC):
    @abstractmethod
    def sign(self, array: Array) -> Array:
        """
        Return the sign of a tensor.

        Args:
            array (Array): The tensor.

        Returns:
            Array: The sign tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def maximum(self, array1: Array, array2: Array) -> Array:
        """
        Element-wise maximum of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise maximum in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        """
        Compute index of maximum value.

        Args:
            array (Array): Input array.
            dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Indices of maximum values.

        """

    @abstractmethod
    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        """
        Compute index of minimum value.

        Args:
            array (Array): Input array.
            dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Indices of minimum values.

        """

    @abstractmethod
    def set_item(
        self,
        array: Array,
        key: ArrayKey,
        value: Array,
    ) -> None:
        """
        Set the item at the specified index of the array to the given value.

        Args:
            array (Array): The tensor.
            key (ArrayKey): The key or index to set.
            value (Array): The value to set.

        Raises:
            TypeError: If the type is not supported.
            NotImplementedError: If the operation is not supported due to immutability.

        """

    @abstractmethod
    def get_item(self, array: Array, key: ArrayKey) -> Array:
        """
        Get the item at the specified index of the array.

        Args:
            array (Array): The tensor.
            key (ArrayKey): The key or index to get.

        Returns:
            Array: The item at the specified index.

        """
