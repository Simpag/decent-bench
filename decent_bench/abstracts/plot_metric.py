from abc import ABC, abstractmethod
from collections.abc import Sequence

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem

from .cost import Cost

X = float
Y = float


class PlotMetric[CF: Cost](ABC):
    """
    Metric to plot at the end of the benchmarking execution.

    Args:
        x_log: whether to apply log scaling to the x-axis.
        y_log: whether to apply log scaling to the y-axis.

    """

    def __init__(self, *, x_log: bool = False, y_log: bool = True):
        self.x_log = x_log
        self.y_log = y_log

    @property
    @abstractmethod
    def plot_description(self) -> str:
        """Label for the y-axis."""

    @abstractmethod
    def get_data_from_trial(
        self,
        agents: list[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> Sequence[tuple[X, Y]]:
        """Extract trial data in the form of (x, y) datapoints."""
