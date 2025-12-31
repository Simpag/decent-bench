from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem

from .cost import Cost

Statistic = Callable[[Sequence[float]], float]


class TableMetric[CF: Cost](ABC):
    """
    Metric to display in the statistical results table at the end of the benchmarking execution.

    Args:
        statistics: sequence of statistics such as :func:`min`, :func:`sum`, and :func:`~numpy.average` used for
            aggregating the data retrieved with :func:`get_data_from_trial` into a single value, each statistic gets its
            own row in the table
        fmt: format string used to format the values in the table, defaults to ".2e". Common formats include:
            - ".2e": scientific notation with 2 decimal places
            - ".3f": fixed-point notation with 3 decimal places
            - ".4g": general format with 4 significant digits
            - ".1%": percentage format with 1 decimal place

            Where the integer specifies the precision.
            See :meth:`str.format` documentation for details on the format string options.

    """

    def __init__(self, statistics: Sequence[Statistic], fmt: str = ".2e"):
        self.statistics = statistics
        self.fmt = fmt

    @property
    @abstractmethod
    def table_description(self) -> str:
        """Metric description to display in the table."""

    @abstractmethod
    def get_data_from_trial(
        self,
        agents: Sequence[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> Sequence[float]:
        """Extract trial data to be aggregated into a single value by each of the *statistics*."""
