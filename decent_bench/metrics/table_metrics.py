import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import tabulate as tb
from numpy import linalg as la
from scipy import stats

import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench.abstracts.algorithm import Algorithm
from decent_bench.abstracts.cost import Cost, FunctionCost, GradientCost
from decent_bench.abstracts.table_metric import TableMetric
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.utils.logger import LOGGER


class Regret(TableMetric[FunctionCost]):
    """
    Global regret using the agents' final x.

    Global regret is defined as:

    .. include:: snippets/global_cost_error.rst
    """

    table_description: str = "regret \n[<1e-9 = exact conv.]"

    def get_data_from_trial[CF_func: FunctionCost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF_func]],
        problem: BenchmarkProblem[CF_func],
    ) -> tuple[float]:
        return (utils.regret(agents, problem, iteration=-1),)


class GradientNorm(TableMetric[GradientCost]):
    """
    Global gradient norm using the agents' final x.

    Global gradient norm is defined as:

    .. include:: snippets/global_gradient_optimality.rst
    """

    table_description: str = "gradient norm"

    def get_data_from_trial[CF_grad: GradientCost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF_grad]],
        _: BenchmarkProblem[CF_grad],
    ) -> tuple[float]:
        return (utils.gradient_norm(agents, iteration=-1),)


class XError(TableMetric[Cost]):
    r"""
    X error per agent as defined below.

    .. math::
        \{ \|\mathbf{x}_i - \mathbf{x}^\star\|, \|\mathbf{x}_j - \mathbf{x}^\star\|, ... \}

    where :math:`\mathbf{x}_i` is agent i's final x,
    :math:`\mathbf{x}_j` is agent j's final x,
    and :math:`\mathbf{x}^\star` is the optimal x defined in the *problem*.

    """

    table_description: str = "x error"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> list[float]:
        return [
            float(la.norm(iop.to_numpy(problem.x_optimal) - iop.to_numpy(a.x_history[max(a.x_history)])))
            for a in agents
        ]


class AsymptoticConvergenceOrder(TableMetric[Cost]):
    """
    Asymptotic convergence order per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """

    table_description: str = "asymptotic convergence order"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> list[float]:
        return [utils.asymptotic_convergence_rate_and_order(a, problem)[1] for a in agents]


class AsymptoticConvergenceRate(TableMetric[Cost]):
    """
    Asymptotic convergence rate per agent as defined below.

    .. include:: snippets/asymptotic_convergence_rate_and_order.rst
    """

    table_description: str = "asymptotic convergence rate"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> list[float]:
        return [utils.asymptotic_convergence_rate_and_order(a, problem)[0] for a in agents]


class IterativeConvergenceOrder(TableMetric[Cost]):
    """
    Iterative convergence order per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """

    table_description: str = "iterative convergence order"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> list[float]:
        return [utils.iterative_convergence_rate_and_order(a, problem)[1] for a in agents]


class IterativeConvergenceRate(TableMetric[Cost]):
    """
    Iterative convergence rate per agent as defined below.

    .. include:: snippets/iterative_convergence_rate_and_order.rst
    """

    table_description: str = "iterative convergence rate"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        problem: BenchmarkProblem[CF],
    ) -> list[float]:
        return [utils.iterative_convergence_rate_and_order(a, problem)[0] for a in agents]


class XUpdates(TableMetric[Cost]):
    """Number of iterations/updates of x per agent."""

    table_description: str = "nr x updates"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_x_updates for a in agents]


class FunctionCalls(TableMetric[Cost]):
    """Number of cost function evaluate calls per agent."""

    table_description: str = "nr function calls"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_function_calls for a in agents]


class GradientCalls(TableMetric[Cost]):
    """Number of cost function gradient calls per agent."""

    table_description: str = "nr gradient calls"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_gradient_calls for a in agents]


class HessianCalls(TableMetric[Cost]):
    """Number of cost function hessian calls per agent."""

    table_description: str = "nr hessian calls"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_hessian_calls for a in agents]


class ProximalCalls(TableMetric[Cost]):
    """Number of cost function proximal calls per agent."""

    table_description: str = "nr proximal calls"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_proximal_calls for a in agents]


class SentMessages(TableMetric[Cost]):
    """Number of sent messages per agent."""

    table_description: str = "nr sent messages"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_sent_messages for a in agents]


class ReceivedMessages(TableMetric[Cost]):
    """Number of received messages per agent."""

    table_description: str = "nr received messages"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_received_messages for a in agents]


class SentMessagesDropped(TableMetric[Cost]):
    """Number of sent messages that were dropped per agent."""

    table_description: str = "nr sent messages dropped"

    def get_data_from_trial[CF: Cost](  # noqa: D102
        self,
        agents: Sequence[AgentMetricsView[CF]],
        _: BenchmarkProblem[CF],
    ) -> list[float]:
        return [a.n_sent_messages_dropped for a in agents]


DEFAULT_TABLE_METRICS: list[TableMetric[Cost] | TableMetric[FunctionCost] | TableMetric[GradientCost]] = [
    Regret([utils.single]),
    GradientNorm([utils.single]),
    XError([min, np.average, max]),
    AsymptoticConvergenceOrder([np.average]),
    AsymptoticConvergenceRate([np.average]),
    IterativeConvergenceOrder([np.average]),
    IterativeConvergenceRate([np.average]),
    XUpdates([np.average, sum]),
    FunctionCalls([np.average, sum]),
    GradientCalls([np.average, sum]),
    HessianCalls([np.average, sum]),
    ProximalCalls([np.average, sum]),
    SentMessages([np.average, sum]),
    ReceivedMessages([np.average, sum]),
    SentMessagesDropped([np.average, sum]),
]
"""
- :class:`Regret` - :func:`~.metric_utils.single`
- :class:`GradientNorm` - :func:`~.metric_utils.single`
- :class:`XError` - :func:`min`, :func:`~numpy.average`, :func:`max`
- :class:`AsymptoticConvergenceOrder` - :func:`~numpy.average`
- :class:`AsymptoticConvergenceRate` - :func:`~numpy.average`
- :class:`IterativeConvergenceOrder` - :func:`~numpy.average`
- :class:`IterativeConvergenceRate` - :func:`~numpy.average`
- :class:`XUpdates` - :func:`~numpy.average`, :func:`sum`
- :class:`FunctionCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`GradientCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`HessianCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`ProximalCalls` - :func:`~numpy.average`, :func:`sum`
- :class:`SentMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`ReceivedMessages` - :func:`~numpy.average`, :func:`sum`
- :class:`SentMessagesDropped` - :func:`~numpy.average`, :func:`sum`

:meta hide-value:
"""


TABLE_METRICS_DOC_LINK = "https://decent-bench.readthedocs.io/en/latest/api/decent_bench.metrics.table_metrics.html"


def tabulate[CF: Cost](
    resulting_agent_states: dict[Algorithm[CF], list[list[AgentMetricsView[CF]]]],
    problem: BenchmarkProblem[CF],
    metrics: Sequence[TableMetric[CF]],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
) -> None:
    """
    Print table with confidence intervals, one row per metric and statistic, and one column per algorithm.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal`,
            are used for metric calculations
        metrics: metrics to calculate
        confidence_level: confidence level of the confidence intervals
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document

    """
    if not metrics:
        return
    LOGGER.info(f"Table metric definitions can be found here: {TABLE_METRICS_DOC_LINK}")
    algs = list(resulting_agent_states)
    headers = ["Metric (statistic)"] + [alg.name for alg in algs]
    rows: list[list[str]] = []
    statistics_abbr = {"average": "avg", "median": "mdn"}
    with warnings.catch_warnings(action="ignore"), utils.MetricProgressBar() as progress:
        n_statistics = sum(len(metric.statistics) for metric in metrics)
        table_task = progress.add_task("Generating table", total=n_statistics, status="")
        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.table_description}")
            data_per_trial = [_data_per_trial(resulting_agent_states[a], problem, metric) for a in algs]
            for statistic in metric.statistics:
                row = [f"{metric.table_description} ({statistics_abbr.get(statistic.__name__) or statistic.__name__})"]
                for i in range(len(algs)):
                    agg_data_per_trial = [statistic(trial) for trial in data_per_trial[i]]
                    mean, margin_of_error = _calculate_mean_and_margin_of_error(agg_data_per_trial, confidence_level)
                    formatted_confidence_interval = _format_confidence_interval(mean, margin_of_error, metric.fmt)
                    row.append(formatted_confidence_interval)
                rows.append(row)
                progress.advance(table_task)
        progress.update(table_task, status="Finalizing table")
    formatted_table = tb.tabulate(rows, headers, tablefmt=table_fmt)
    LOGGER.info("\n" + formatted_table)


def _data_per_trial(
    agents_per_trial: list[list[AgentMetricsView[Any]]],
    problem: BenchmarkProblem[Any],
    metric: TableMetric[Any],
) -> list[Sequence[float]]:
    data_per_trial: list[Sequence[float]] = []
    for agents in agents_per_trial:
        trial_data = metric.get_data_from_trial(agents, problem)
        data_per_trial.append(trial_data)

    return data_per_trial


def _calculate_mean_and_margin_of_error(data: list[float], confidence_level: float) -> tuple[float, float]:
    mean = np.mean(data)
    sem = stats.sem(data) if len(set(data)) > 1 else None
    raw_interval = (
        stats.t.interval(confidence=confidence_level, df=len(data) - 1, loc=mean, scale=sem) if sem else (mean, mean)
    )
    if np.isfinite(mean) and np.isfinite(raw_interval).all():
        return (float(mean), float(mean - raw_interval[0]))

    return np.nan, np.nan


def _format_confidence_interval(mean: float, margin_of_error: float, fmt: str) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    formatted_confidence_interval = f"{mean:{fmt}} \u00b1 {margin_of_error:{fmt}}"

    if any(np.isnan([mean, margin_of_error])):
        formatted_confidence_interval += " (diverged?)"

    return formatted_confidence_interval


def _is_valid_float_format_spec(fmt: str) -> bool:
    """
    Validate that the given format spec can be used to format a float.

    This avoids attempting to format real values with an invalid format string.

    """
    try:
        f"{0.01:{fmt}}"
    except (ValueError, TypeError):
        return False
    return True
