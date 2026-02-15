import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import tabulate as tb
from scipy import stats

import decent_bench.metrics.metric_utils as utils
from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics._metric import Metric
from decent_bench.utils.logger import LOGGER

STATISTICS_ABBR = {"average": "avg", "median": "mdn"}


def create_tables(
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]],
    problem: BenchmarkProblem,
    metrics: list[Metric],
    confidence_level: float,
    table_fmt: Literal["grid", "latex"],
    *,
    table_path: Path | None = None,
) -> dict[Algorithm, dict[Metric, dict[str, tuple[float, float]]]] | None:
    """
    Print table with confidence intervals, one row per metric and statistic, and one column per algorithm.

    Args:
        resulting_agent_states: resulting agent states from the trial executions, grouped by algorithm
        problem: benchmark problem whose properties, e.g.
            :attr:`~decent_bench.benchmark_problem.BenchmarkProblem.x_optimal`,
            are used for metric calculations
        metrics: metrics to calculate
        confidence_level: confidence level for computing confidence intervals of the metrics, expressed as a value
            between 0 and 1 (e.g., 0.95 for 95% confidence, 0.99 for 99% confidence). Higher values result in
            wider confidence intervals.
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        table_path: optional path to save the table as a text file, if not provided the table is not saved to a file

    Returns:
        A nested dictionary containing the mean and margin of error for each metric and statistic, structured as
        {algorithm: {metric: {statistic_name: (mean, margin_of_error)}}}

    """
    if not metrics:
        return None
    results: dict[Algorithm, dict[Metric, dict[str, tuple[float, float]]]] = {
        a: {m: {} for m in metrics} for a in resulting_agent_states
    }
    algs = list(resulting_agent_states)
    headers = ["Metric (statistic)"] + [alg.name for alg in algs]
    rows: list[list[str]] = []
    with warnings.catch_warnings(action="ignore"), utils.MetricProgressBar() as progress:
        table_task = progress.add_task(
            "Generating table",
            total=sum(len(metric.statistics) for metric in metrics),
            status="",
        )
        for metric in metrics:
            progress.update(table_task, status=f"Task: {metric.table_description}")
            data_per_trial = [_table_data_per_trial(resulting_agent_states[a], problem, metric) for a in algs]
            for statistic in metric.statistics:
                row = [f"{metric.table_description} ({STATISTICS_ABBR.get(statistic.__name__) or statistic.__name__})"]
                for i in range(len(algs)):
                    agg_data_per_trial = [statistic(trial) for trial in data_per_trial[i]]
                    mean, margin_of_error = _calculate_mean_and_margin_of_error(agg_data_per_trial, confidence_level)
                    formatted_confidence_interval = _format_confidence_interval(
                        mean,
                        margin_of_error,
                        metric.fmt,
                        metric.can_diverge,
                    )
                    results[algs[i]][metric][STATISTICS_ABBR.get(statistic.__name__) or statistic.__name__] = (
                        mean,
                        margin_of_error,
                    )
                    row.append(formatted_confidence_interval)
                rows.append(row)
                progress.advance(table_task)
        progress.update(table_task, status="Finalizing table")
    grid_table = tb.tabulate(rows, headers, tablefmt="grid")
    latex_table = tb.tabulate(rows, headers, tablefmt="latex")
    LOGGER.info("\n" + latex_table if table_fmt == "latex" else "\n" + grid_table)
    if table_path:
        # Save both latex and grid tables to two files with appropriate suffixes
        table_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path = table_path.parent / (table_path.stem + ".tex")
        grid_path = table_path.parent / (table_path.stem + ".txt")
        latex_path.write_text(latex_table, encoding="utf-8")
        grid_path.write_text(grid_table, encoding="utf-8")
        LOGGER.info(f"Saved LaTeX table to {latex_path}")
        LOGGER.info(f"Saved grid table to {grid_path}")

    return results


def _table_data_per_trial(
    agents_per_trial: list[list[AgentMetricsView]],
    problem: BenchmarkProblem,
    metric: Metric,
) -> list[Sequence[float]]:
    data_per_trial: list[Sequence[float]] = []
    for agents in agents_per_trial:
        trial_data = metric.get_table_data(agents, problem)
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


def _format_confidence_interval(mean: float, margin_of_error: float, fmt: str, can_diverge: bool) -> str:
    if not _is_valid_float_format_spec(fmt):
        LOGGER.warning(f"Invalid format string '{fmt}', defaulting to scientific notation")
        fmt = ".2e"

    formatted_confidence_interval = f"{mean:{fmt}} \u00b1 {margin_of_error:{fmt}}"

    if any(np.isnan([mean, margin_of_error])) and can_diverge:
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
