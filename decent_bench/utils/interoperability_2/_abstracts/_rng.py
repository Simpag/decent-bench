from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array


class _BackendRng(ABC):
    @abstractmethod
    def normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        shape: tuple[int, ...] = (),
    ) -> Array:
        """
        Create an array of random values with the specified shape and framework.

        Values are drawn from a normal distribution with mean `mean` and standard deviation `std`.

        Args:
            mean (float): Mean of the normal distribution.
            std (float): Standard deviation of the normal distribution.
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Array: Array of normally distributed random values.

        """

    @abstractmethod
    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] = (),
    ) -> Array:
        """
        Create an array of random values with the specified shape and framework.

        Values are drawn uniformly from [low, high).

        Args:
            low (float): Lower bound of the uniform distribution.
            high (float): Upper bound of the uniform distribution.
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Array: Array of uniformly distributed random values

        """

    @abstractmethod
    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        """
        Create an array of random values with the same shape and type as the input.

        Values are drawn from a normal distribution with mean `mean` and standard deviation `std`.

        Args:
            array (Array): Input array.
            mean (float): Mean of the normal distribution.
            std (float): Standard deviation of the normal distribution.

        Returns:
            Array: Array of normally distributed random values.

        """

    @abstractmethod
    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        """
        Create an array of random values with the same shape and type as the input.

        Values are drawn uniformly from [low, high).

        Args:
            array (Array): Input array.
            low (float): Lower bound of the uniform distribution.
            high (float): Upper bound of the uniform distribution.

        Returns:
            Array: Array of uniformly distributed random values.

        """

    @abstractmethod
    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        """
        Randomly sample elements from an array.

        Args:
            array (Array): Input array to sample from.
            size (int): Number of samples to draw.
            replace (bool): Whether to sample with replacement.

        Returns:
            Array: Sampled values.

        """
