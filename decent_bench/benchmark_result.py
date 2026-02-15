from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics._metric import Metric
from decent_bench.networks import Network


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark execution, containing the results and metadata.

    This class is used to store the results and metadata of a benchmark execution.
    It is returned by the :func:`~decent_bench.benchmark.benchmark` function and contains
    all the information about the benchmark run, including the problem definition,
    algorithm states, table results, and plot results.

    * `benchmark_problem`: contains the definition of the benchmark problem that was executed.
    * `states`: contains the final states of the algorithms after execution, organized by algorithm where
      each algorithm maps to a sequence of network states (one per trial).
    * `table_results`: contains the table results for each algorithm and metric, organized by algorithm and metric.
    * `plot_results`: contains the plot results for each algorithm and metric, organized by algorithm and metric,
      where each metric maps to a tuple of sequences representing (x, y_mean, y_min, y_max) for plotting.

    These results can be used for analysis, visualization, and comparison of the algorithms' performance on the
    benchmark problem. It is especially useful for hyperparameter tuning, algorithm comparison, and understanding
    the behavior of algorithms under different conditions.
    """

    benchmark_problem: BenchmarkProblem
    states: Mapping[Algorithm, Sequence[Network]]
    table_results: Mapping[Algorithm, Mapping[Metric, Mapping[str, tuple[float, float]]]] | None
    plot_results: (
        Mapping[Algorithm, Mapping[Metric, tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]]]
        | None
    )
