from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, final

from decent_bench.networks import P2PNetwork
from decent_bench.utils.types import CF_contra


class Algorithm(ABC, Generic[CF_contra]):  # noqa: UP046
    """Distributed algorithm - agents collaborate to solve an optimization problem using peer-to-peer communication."""

    @property
    @abstractmethod
    def iterations(self) -> int:
        """Number of iterations to run the algorithm for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, network: P2PNetwork[CF_contra]) -> None:
        """
        Initialize the algorithm.

        Args:
            network: provides agents, neighbors etc.

        """

    @abstractmethod
    def step(self, network: P2PNetwork[CF_contra], iteration: int) -> None:
        """
        Perform one iteration of the algorithm.

        Args:
            network: provides agents, neighbors etc.
            iteration: current iteration number

        """

    def finalize(self, network: P2PNetwork[CF_contra]) -> None:
        """
        Finalize the algorithm.

        Note:
            Override method as needed.
            Does not need to be implemented if no finalization is required.
            By default it is used to clean up auxiliary variables to free memory.

        Args:
            network: provides agents, neighbors etc.

        """
        for i in network.agents():
            if i.aux_vars is not None:
                i.aux_vars.clear()

    @final
    def run(self, network: P2PNetwork[CF_contra], progress_callback: Callable[[int], None] | None = None) -> None:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations
            and finally :meth:`finalize`.

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Args:
            network: provides agents, neighbors etc.
            progress_callback: optional callback to report progress after each iteration.

        """
        self.initialize(network)
        for k in range(self.iterations):
            self.step(network, k)
            if progress_callback is not None:
                progress_callback(k)
        self.finalize(network)
