from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array


class _BackendLinalg(ABC):
    @abstractmethod
    def dot(self, array1: Array, array2: Array) -> Array:
        """
        Dot product of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of the dot product in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def matmul(self, array1: Array, array2: Array) -> Array:
        """
        Matrix multiplication of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of matrix multiplication in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute the norm of an array.

        Args:
            array (Array): The tensor.
            p (float): The order of the norm.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute the norm.
                If None, computes norm over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: The norm of the tensor.

        Raises:
            TypeError: If the type is not supported.

        """
