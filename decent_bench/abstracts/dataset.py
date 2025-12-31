from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from decent_bench.utils.array import Array

DatasetPartition: TypeAlias = tuple[Array, Array]  # noqa: UP040
"""Tuple of (A, b) representing one dataset partition."""


class Dataset(ABC):
    """Dataset containing partitions in the form of feature matrix A and target vector b."""

    @abstractmethod
    def training_partitions(self) -> Sequence[DatasetPartition]:
        """Partitions used for finding the optimal optimization variable x."""
