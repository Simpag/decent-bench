import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from logging.handlers import QueueListener
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from rich.status import Status

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.costs import Cost
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import plot_metrics as pm
from decent_bench.metrics import table_metrics as tm
from decent_bench.metrics.plot_metrics import PlotMetric
from decent_bench.metrics.table_metrics import TableMetric
from decent_bench.networks import P2PNetwork, create_distributed_network
from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.progress_bar import ProgressBarController
from decent_bench.utils.types import CF

if TYPE_CHECKING:
    from decent_bench.utils.progress_bar import ProgressBarHandle


def benchmark(
    benchmark_problem: BenchmarkProblem[CF],
    algorithms: tuple[Algorithm[CF], ...],
    plot_metrics: tuple[PlotMetric[CF], ...] | None = None,
    table_metrics: tuple[TableMetric[CF], ...] | None = None,
    *,
    table_fmt: Literal["grid", "latex"] = "grid",
    plot_grid: bool = True,
    plot_path: str | None = None,
    computational_cost: pm.ComputationalCost | None = None,
    x_axis_scaling: float = 1e-4,
    n_trials: int = 30,
    confidence_level: float = 0.95,
    log_level: int = logging.INFO,
    max_processes: int | None = None,
    progress_step: int | None = None,
    show_speed: bool = False,
    show_trial: bool = False,
    compare_iterations_and_computational_cost: bool = False,
) -> None:
    """
    Benchmark distributed algorithms.

    Args:
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints
        algorithms: algorithms to benchmark
        plot_metrics: metrics to plot after the execution, defaults to
            :const:`~decent_bench.metrics.plot_metrics.DEFAULT_PLOT_METRICS`
        table_metrics: metrics to tabulate as confidence intervals after the execution, defaults to
            :const:`~decent_bench.metrics.table_metrics.DEFAULT_TABLE_METRICS`
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        plot_grid: whether to show grid lines on the plots
        plot_path: optional file path to save the generated plot as an image file (e.g., "plots.png"). If ``None``,
            the plot will only be displayed
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        x_axis_scaling: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
        confidence_level: confidence level of the confidence intervals
        log_level: minimum level to log, e.g. :data:`logging.INFO`
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default. If your algorithm is very lightweight you
            may want to set this to 1 to avoid the multiprocessing overhead.
        progress_step: if provided, the progress bar will step every `progress_step` iterations.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.
        show_speed: whether to show speed (iterations/second) in the progress bar.
        show_trial: whether to show which trials are currently running in the progress bar.
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided.

    Note:
        If ``progress_step`` is too small performance may degrade due to the
        overhead of updating the progress bar too often.

        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        .. include:: snippets/computational_cost.rst

        If ``computational_cost`` is provided and ``compare_iterations_and_computational_cost`` is ``True``, each metric
        will be plotted twice: once against computational cost and once against iterations.
        Computational cost plots will be shown on the left and iteration plots on the right.

        ``benchmark_problem``, `algorithms`, ``plot_metrics``, and ``table_metrics`` must be compatible in terms of
        the cost function type they support. ``benchmark_problem`` defines the cost function type, and the algorithms
        and metrics must support that type. For example, if ``benchmark_problem`` provides ``GradientCost``, all
        algorithms and metrics must support ``GradientCost`` and its supertypes.

    """
    _ensure_benchmark_compatibility(benchmark_problem, algorithms, plot_metrics, table_metrics)
    manager = Manager()
    log_listener = logger.start_log_listener(manager, log_level)
    LOGGER.info("Starting benchmark execution ")
    with Status("Generating initial network state"):
        nw_init_state = create_distributed_network(benchmark_problem)
    LOGGER.debug(f"Nr of agents: {len(nw_init_state.agents())}")
    prog_ctrl = ProgressBarController(manager, algorithms, n_trials, progress_step, show_speed, show_trial)
    resulting_nw_states = _run_trials(algorithms, n_trials, nw_init_state, prog_ctrl, log_listener, max_processes)
    LOGGER.info("All trials complete")
    resulting_agent_states: dict[Algorithm[CF], list[list[AgentMetricsView[CF]]]] = {}
    for alg, networks in resulting_nw_states.items():
        resulting_agent_states[alg] = [[AgentMetricsView.from_agent(a) for a in nw.agents()] for nw in networks]
    if table_metrics:
        tm.tabulate(resulting_agent_states, benchmark_problem, table_metrics, confidence_level, table_fmt)
    if plot_metrics:
        pm.plot(
            resulting_agent_states,
            benchmark_problem,
            plot_metrics,
            computational_cost,
            x_axis_scaling,
            compare_iterations_and_computational_cost,
            plot_path,
            plot_grid,
        )
    LOGGER.info("Benchmark execution complete, thanks for using decent-bench")
    log_listener.stop()


def _run_trials(  # noqa: PLR0917
    algorithms: tuple[Algorithm[Any], ...],
    n_trials: int,
    nw_init_state: P2PNetwork[Any],
    progress_bar_ctrl: ProgressBarController,
    log_listener: QueueListener,
    max_processes: int | None,
) -> dict[Algorithm[Any], list[P2PNetwork[Any]]]:
    progress_bar_handle = progress_bar_ctrl.get_handle()
    if max_processes == 1:
        result = {
            alg: [_run_trial(alg, nw_init_state, progress_bar_handle, trial) for trial in range(n_trials)]
            for alg in algorithms
        }
    else:
        with ProcessPoolExecutor(
            initializer=logger.start_queue_logger, initargs=(log_listener.queue,), max_workers=max_processes
        ) as executor:
            LOGGER.debug(f"Concurrent processes: {executor._max_workers}")  # type: ignore[attr-defined] # noqa: SLF001
            all_futures = {
                alg: [
                    executor.submit(_run_trial, alg, nw_init_state, progress_bar_handle, trial)
                    for trial in range(n_trials)
                ]
                for alg in algorithms
            }
            result = {alg: [f.result() for f in as_completed(futures)] for alg, futures in all_futures.items()}

    progress_bar_ctrl.stop()
    return result


def _run_trial(
    algorithm: Algorithm[Any],
    nw_init_state: P2PNetwork[Any],
    progress_bar_handle: "ProgressBarHandle",
    trial: int,
) -> P2PNetwork[Any]:
    progress_bar_handle.start_progress_bar(algorithm, trial)
    network = deepcopy(nw_init_state)
    alg = deepcopy(algorithm)

    with warnings.catch_warnings(action="error"):
        try:
            alg.run(network, lambda iteration: progress_bar_handle.advance_progress_bar(algorithm, iteration))
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")
    return network


def _ensure_benchmark_compatibility(
    benchmark_problem: BenchmarkProblem[Any],
    algorithms: tuple[Algorithm[Any], ...],
    plot_metrics: tuple[PlotMetric[Any], ...] | None,
    table_metrics: tuple[TableMetric[Any], ...] | None,
) -> None:
    """
    Raise a helpful error if algorithms/metrics are incompatible with costs.

    ``benchmark_problem`` defines the concrete cost type used by all agents.
    Each algorithm and metric is parameterized by the *least* cost interface it
    requires (e.g. ``GradientCost``). At runtime we check that this requirement
    is satisfied by the benchmark's cost type using ``isinstance`` against the
    inferred protocol.

    Raises:
        TypeError: if any algorithm or metric is incompatible with the costs

    """
    costs = list(benchmark_problem.costs)
    if not costs:
        return

    def _check(obj: Any, role: str, generic_base: type[Any]) -> None:  # noqa: ANN401
        required = _get_required_cost_type(obj, generic_base)
        if required is None:
            return
        incompatible = [c for c in costs if not isinstance(c, required)]
        if incompatible:
            offending_type = type(incompatible[0]).__name__
            try:
                offending_super_type = type(incompatible[0]).__bases__[0].__name__
            except Exception:
                offending_super_type = "Unknown"
            raise TypeError(
                f"Incompatible {role}(s): {type(obj).__name__} requires costs implementing {required.__name__}, "
                f"but benchmark_problem includes cost of type {offending_type} which implements {offending_super_type}."
            )

    for alg in algorithms:
        _check(alg, "algorithm", Algorithm)

    if table_metrics is not None:
        for table_metric in table_metrics:
            if not isinstance(table_metric, TableMetric):
                raise TypeError(f"Expected TableMetric, got {type(table_metric).__name__}")
            _check(table_metric, "table metric", TableMetric)

    if plot_metrics is not None:
        for plot_metric in plot_metrics:
            if not isinstance(plot_metric, PlotMetric):
                raise TypeError(f"Expected PlotMetric, got {type(plot_metric).__name__}")
            _check(plot_metric, "plot metric", PlotMetric)


def _get_required_cost_type(obj: Any, generic_base: type[Any]) -> type[Cost] | None:  # noqa: ANN401
    """
    Infer the cost protocol required by a generic base class at runtime.

    This inspects ``__orig_bases__`` on the class hierarchy to find the type
    argument used with ``generic_base`` (e.g. ``Algorithm[GradientCost]``).

    It returns ``None`` if no such base is found or if the argument is not a
    concrete runtime-checkable protocol or class.
    """
    cls = type(obj)
    for base_cls in cls.mro():
        for base in getattr(base_cls, "__orig_bases__", ()):
            origin = get_origin(base) or base
            if origin is generic_base:
                args = get_args(base)
                if args and isinstance(args[0], type):
                    return args[0]
    return None
