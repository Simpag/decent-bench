from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices


class _BackendArrayCreation(ABC):
    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Array:
        """
        Create an array of zeros.

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
        Create an array of ones.

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
        Create an n x n identity matrix.

        Args:
            n (int): Size of the identity matrix.

        Returns:
            Array: Identity matrix.

        """

    @abstractmethod
    def eye_like(self, array: Array) -> Array:
        """
        Create an identity matrix with the same shape as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Identity matrix.

        """

    @abstractmethod
    def device_to_native(self, device: SupportedDevices) -> Any:  # noqa: ANN401
        """
        Convert :class:`SupportedDevices` to the backend's native device representation.

        Args:
            device (SupportedDevices): Device.

        Returns:
            Any: Backend-native device representation.

        """

    @abstractmethod
    def device_of(self, array: Array) -> SupportedDevices:
        """
        Return the :class:`SupportedDevices` of the given array.

        The array is assumed to belong to this backend's framework.

        Args:
            array (Array): Input array.

        Returns:
            SupportedDevices: Device the array lives on.

        """
