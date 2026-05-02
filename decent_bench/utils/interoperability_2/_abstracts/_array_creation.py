from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedArrayTypes, SupportedDevices, SupportedFrameworks


class _BackendArrayCreation(ABC):
    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Array:
        """
        Create a Array of zeros.

        Args:
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Array: Array of zeros.

        """

    @abstractmethod
    def zeros_like(self, array: Array) -> Array:
        """
        Create an array of zeros with the same shape and type as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Array of zeros in the same framework type as the input.

        """

    @abstractmethod
    def ones(self, shape: tuple[int, ...]) -> Array:
        """
        Create a Array of ones.

        Args:
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Array: Array of ones.

        """

    @abstractmethod
    def ones_like(self, array: Array) -> Array:
        """
        Create an array of ones with the same shape and type as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Array of ones in the same framework type as the input.

        """

    @abstractmethod
    def eye(self, n: int) -> Array:
        """
        Create an identity matrix of size n x n in the specified framework.

        Args:
            n (int): Size of the identity matrix.

        Returns:
            Array: Identity matrix in the specified framework type.

        """

    @abstractmethod
    def eye_like(self, array: Array) -> Array:
        """
        Create an identity matrix with the same shape as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Identity matrix in the same framework type as the input.

        """

    @abstractmethod
    def device_to_framework_device(self, device: SupportedDevices, framework: SupportedFrameworks) -> Any:  # noqa: ANN401
        """
        Convert SupportedDevices literal to framework-specific device representation.

        Args:
            device (SupportedDevices): Device literal ("cpu" or "gpu").
            framework (SupportedFrameworks): Framework literal ("numpy", "torch", "tensorflow", "jax").

        Returns:
            Any: Framework-specific device representation.

        Raises:
            ValueError: If the framework is unsupported.

        """

    @abstractmethod
    def framework_device_of_array(self, array: Array) -> tuple[SupportedFrameworks, SupportedDevices]:
        """
        Determine the framework and device of the given Array.

        Args:
            array (Array): Input array.

        Returns:
            tuple[SupportedFrameworks, SupportedDevices]: Framework and device of the array.

        Raises:
            TypeError: if the framework type of `array` is unsupported.

        """
