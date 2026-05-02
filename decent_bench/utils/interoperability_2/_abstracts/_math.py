from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array


class _BackendMath(ABC):
    @abstractmethod
    def sum(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Sum elements of an array.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
                If None, sums over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Summed value.

        """

    @abstractmethod
    def mean(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute mean of array elements.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute the mean.
                If None, computes mean of flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Mean value.

        """

    @abstractmethod
    def min(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute minimum of array elements.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
                If None, finds minimum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Minimum value.

        """

    @abstractmethod
    def max(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute maximum of array elements.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
                If None, finds maximum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Maximum value.

        """

    @abstractmethod
    def add(self, array1: Array, array2: Array) -> Array:
        """
        Element-wise addition of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise addition in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def iadd[T: Array](self, array1: T, array2: Array) -> T:
        """
        Element-wise in-place addition of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise in-place addition in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def sub(self, array1: Array, array2: Array) -> Array:
        """
        Element-wise subtraction of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise subtraction in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def isub[T: Array](self, array1: T, array2: Array) -> T:
        """
        Element-wise in-place subtraction of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise in-place subtraction in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def mul(self, array1: Array, array2: Array) -> Array:
        """
        Element-wise multiplication of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise multiplication in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def imul[T: Array](self, array1: T, array2: Array) -> T:
        """
        Element-wise in-place multiplication of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise in-place multiplication in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def div(self, array1: Array, array2: Array) -> Array:
        """
        Element-wise division of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise division in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def idiv[T: Array](self, array1: T, array2: Array) -> T:
        """
        Element-wise in-place division of two arrays.

        Args:
            array1 (Array): First input array.
            array2 (Array): Second input array.

        Returns:
            Array: Result of element-wise in-place division in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def pow(self, array: Array, p: float) -> Array:
        """
        Raise array to p power.

        Args:
            array (Array): The tensor.
            p (float): The power.

        Returns:
            Array: The result of the operation.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def ipow[T: Array](self, array: T, p: float) -> T:
        """
        Element-wise in-place power of an array.

        Args:
            array (Array): Input array.
            p (float): The power.

        Returns:
            Array: Result of element-wise in-place power in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported

        """

    @abstractmethod
    def negative(self, array: Array) -> Array:
        """
        Negate array.

        Args:
            array (Array): The tensor.

        Returns:
            Array: The negated tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def absolute(self, array: Array) -> Array:
        """
        Return the absolute value of a tensor.

        Args:
            array (Array): The tensor.

        Returns:
            Array: The absolute value tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def sqrt(self, array: Array) -> Array:
        """
        Return the square root of a tensor.

        Args:
            array (Array): The tensor.

        Returns:
            Array: The square root tensor.

        Raises:
            TypeError: If the type is not supported.

        """
