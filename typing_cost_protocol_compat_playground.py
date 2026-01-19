# ruff: noqa: I001

"""
Static typing playground: cost-protocol compatibility via variance.

This is intentionally a small, standalone file (not a pytest test).

Goal
----
- The benchmark problem "provides" costs of type `C`.
- Algorithms/metrics "require" some cost protocol `R`.
- Valid iff the provided cost is a subtype of the required protocol: `C <: R`.

Mechanism
---------
- `BenchmarkProblem[C]` is *covariant* in `C` (it produces/provides `C`).
- `Algorithm[R]` / metrics are *contravariant* in `R` (they consume/require `R`).

How to use
----------
- Run a type checker on just this file while iterating:
  - `python -m mypy typing_cost_protocol_compat_playground.py`
  - or `pyright typing_cost_protocol_compat_playground.py`

Expected results
----------------
- Lines marked "OK" should type-check.
- Lines marked "E:" should be flagged by the type checker.

Note: This uses the classic `TypeVar(...)` + `Generic[...]` syntax with explicit
variance, which is the most reliable choice across both mypy and pyright today.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from abc import ABC, abstractmethod


# ----- Cost protocol lattice (minimal) -----


class Cost(Protocol):
    """Base cost protocol (placeholder for the playground)."""


class FunctionCost(Cost, Protocol):
    """Cost that supports `function()` evaluation."""

    def function(self) -> float:
        """Evaluate the cost's scalar objective."""
        ...


class GradientCost(Cost, Protocol):
    """Cost that supports `gradient()` evaluation."""

    def gradient(self) -> float:
        """Evaluate the cost's (placeholder) gradient."""
        ...


class HessianCost(Cost, Protocol):
    """Cost that supports `hessian()` evaluation."""

    def hessian(self) -> float:
        """Evaluate the cost's (placeholder) hessian."""
        ...


class ProximalCost(Cost, Protocol):
    """Cost that supports `proximal()` evaluation."""

    def proximal(self) -> float:
        """Evaluate the cost's (placeholder) proximal operator."""
        ...


class FunctionGradientCost(FunctionCost, GradientCost, Protocol):
    """Cost that supports both `function()` and `gradient()`."""


class FunctionGradientHessianProximalCost(
    FunctionGradientCost,
    HessianCost,
    ProximalCost,
    Protocol,
):
    """Cost that supports `function()`, `gradient()`, `hessian()`, and `proximal()`."""


# ----- Generic containers/roles -----
# Type parameter for the benchmark() function and benchmark problem.
# Note: this is intentionally invariant because `BenchmarkProblem.cost` is a
# mutable attribute, and type checkers reject covariant TypeVars in that case.
C = TypeVar("C", bound=Cost)

# Algorithms/metrics *require* a cost => contravariant.
R_contra = TypeVar("R_contra", bound=Cost, contravariant=True)
R_co = TypeVar("R_co", bound=Cost, covariant=True)


@dataclass(eq=False)
class BenchmarkProblem(Generic[C]):  # noqa: UP046
    """Benchmark problem that *provides* a cost of type `C`."""

    cost: C


class Network[CF: Cost]:  # noqa: B903
    """Network that uses a cost of type `CF`."""

    def __init__(self, cost: CF) -> None:
        self.cost = cost


class Algorithm(Protocol[R_contra]):
    """Algorithm that *requires* a cost implementing `R_contra`."""

    def run(self, cost: R_contra) -> None:
        """Run the algorithm against a cost that provides at least `R_contra`."""
        ...


# class Algorithm(ABC, Generic[R_co]):  # noqa: UP046
#     """Algorithm that *requires* a cost implementing `R_contra`."""

#     def _cost_requirement(self) -> type[R_co]:
#         """The cost protocol required by this algorithm."""
#         raise NotImplementedError

#     @abstractmethod
#     def run(self, cost: Network[R_co]) -> None:
#         """Run the algorithm against a cost that provides at least `R_contra`."""


class PlotMetric(Protocol[R_contra]):
    """Plot metric that *requires* a cost implementing `R_contra`."""

    def compute(self, cost: R_contra) -> float:
        """Compute a scalar value given a cost."""
        ...


class TableMetric(Protocol[R_contra]):
    """Table metric that *requires* a cost implementing `R_contra`."""

    def compute(self, cost: R_contra) -> float:
        """Compute a scalar value given a cost."""
        ...


def benchmark(
    *,
    problem: BenchmarkProblem[C],
    algorithms: list[Algorithm[C]],
    plot_metrics: list[PlotMetric[C]],
    table_metrics: list[TableMetric[C]],
) -> None:
    """Validate typing relationships between problem and consumers."""
    _ = plot_metrics
    _ = table_metrics
    for alg in algorithms:
        alg.run(problem.cost)


# ----- Concrete implementations for checking -----


class _FG:
    def function(self) -> float:
        return 0.0

    def gradient(self) -> float:
        return 0.0


class _OnlyFunction:
    def function(self) -> float:
        return 0.0


class _LinearRegressionCost:
    def function(self) -> float:
        return 0.0

    def gradient(self) -> float:
        return 0.0

    def hessian(self) -> float:
        return 0.0

    def proximal(self) -> float:
        return 0.0


# class AlgoNeedsFunction(Algorithm[FunctionCost]):
#     """Algorithm that requires `FunctionCost` (needs `.function`)."""

#     def run(self, cost: Network[FunctionCost]) -> None:
#         """Run with any cost that has `.function`."""
#         _ = cost.cost.function()


# class AlgoNeedsGradient(Algorithm[GradientCost]):
#     """Algorithm that requires `GradientCost` (needs `.gradient`)."""

#     def run(self, cost: Network[GradientCost]) -> None:
#         """Run with any cost that has `.gradient`."""
#         _ = cost.cost.gradient()


# class AlgoNeedsProximal(Algorithm[ProximalCost]):
#     """Algorithm that requires `ProximalCost` (needs `.proximal`)."""

#     def run(self, cost: Network[ProximalCost]) -> None:
#         """Run with any cost that has `.proximal`."""
#         _ = cost.cost.proximal()


# class AlgoNeedsFunctionGradient(Algorithm[FunctionGradientCost]):
#     """Algorithm that requires `FunctionGradientCost` (needs both)."""

#     def run(self, cost: Network[FunctionGradientCost]) -> None:
#         """Run with any cost that has both `.function` and `.gradient`."""
#         _ = cost.cost.function()
#         _ = cost.cost.gradient()


# class AlgoNeedsFunctionGradientHessianProximal(
#     Algorithm[FunctionGradientHessianProximalCost]
# ):
#     """Algorithm that requires `FunctionGradientHessianProximalCost` (needs all four)."""

#     def run(self, cost: Network[FunctionGradientHessianProximalCost]) -> None:
#         """Run with any cost that has `.function`, `.gradient`, `.hessian`, and `.proximal`."""
#         _ = cost.cost.function()
#         _ = cost.cost.gradient()
#         _ = cost.cost.hessian()
#         _ = cost.cost.proximal()


class AlgoNeedsFunction:
    """Algorithm that requires `FunctionCost` (needs `.function`)."""

    def run(self, cost: FunctionCost) -> None:
        """Run with any cost that has `.function`."""
        _ = cost.function()


class AlgoNeedsGradient:
    """Algorithm that requires `GradientCost` (needs `.gradient`)."""

    def run(self, cost: GradientCost) -> None:
        """Run with any cost that has `.gradient`."""
        _ = cost.gradient()


class AlgoNeedsProximal:
    """Algorithm that requires `ProximalCost` (needs `.proximal`)."""

    def run(self, cost: ProximalCost) -> None:
        """Run with any cost that has `.proximal`."""
        _ = cost.proximal()


class AlgoNeedsFunctionGradient:
    """Algorithm that requires `FunctionGradientCost` (needs both)."""

    def run(self, cost: FunctionGradientCost) -> None:
        """Run with any cost that has both `.function` and `.gradient`."""
        _ = cost.function()
        _ = cost.gradient()


class AlgoNeedsFunctionGradientHessianProximal:
    """Algorithm that requires `FunctionGradientHessianProximalCost` (needs all four)."""

    def run(self, cost: FunctionGradientHessianProximalCost) -> None:
        """Run with any cost that has `.function`, `.gradient`, `.hessian`, and `.proximal`."""
        _ = cost.function()
        _ = cost.gradient()
        _ = cost.hessian()
        _ = cost.proximal()


class PlotNeedsCost:
    """Metric that requires only `Cost` (the top protocol)."""

    def compute(self, cost: Cost) -> float:
        """Compute a dummy value."""
        _ = cost
        return 0.0


class PlotNeedsFunction:
    """Metric that requires `FunctionCost`."""

    def compute(self, cost: FunctionCost) -> float:
        """Compute a value from `.function`."""
        return cost.function()


class PlotNeedsGradient:
    """Metric that requires `GradientCost`."""

    def compute(self, cost: GradientCost) -> float:
        """Compute a value from `.gradient`."""
        return cost.gradient()


class TableNeedsGradient:
    """Metric that requires `GradientCost`."""

    def compute(self, cost: GradientCost) -> float:
        """Compute a value from `.gradient`."""
        return cost.gradient()


class TableNeedsFunction:
    """Metric that requires `FunctionCost`."""

    def compute(self, cost: FunctionCost) -> float:
        """Compute a value from `.function`."""
        return cost.function()


# ----- The actual "checks" (type-checker should reason about these) -----


def _typechecks_ok() -> None:
    # Problem provides FunctionGradientCost.
    prob_fg = BenchmarkProblem(_FG())

    # OK: these require *less* than FunctionGradientCost.
    benchmark(
        problem=prob_fg,
        algorithms=[
            AlgoNeedsFunction(),
            AlgoNeedsGradient(),
            AlgoNeedsFunctionGradient(),
        ],
        plot_metrics=[PlotNeedsCost(), PlotNeedsFunction()],
        table_metrics=[TableNeedsGradient()],
    )


def _typechecks_ok_2() -> None:
    # Problem provides only FunctionCost.
    prob_f = BenchmarkProblem(_OnlyFunction())

    # OK: these require <= FunctionCost.
    benchmark(
        problem=prob_f,
        algorithms=[AlgoNeedsFunction(), AlgoNeedsFunction(), AlgoNeedsFunction()],
        plot_metrics=[PlotNeedsCost(), PlotNeedsFunction()],
        table_metrics=[TableNeedsFunction(), TableNeedsFunction()],
    )


def _typechecks_ok_3() -> None:
    # Problem provides only FunctionCost.
    prob_f = BenchmarkProblem(_LinearRegressionCost())

    # OK: these require <= FunctionCost.
    benchmark(
        problem=prob_f,
        algorithms=[
            AlgoNeedsFunction(),
            AlgoNeedsGradient(),
            AlgoNeedsProximal(),
            AlgoNeedsFunctionGradient(),
            AlgoNeedsFunctionGradientHessianProximal(),
        ],
        plot_metrics=[PlotNeedsCost(), PlotNeedsFunction()],
        table_metrics=[TableNeedsFunction(), TableNeedsFunction()],
    )


def _typechecks_errors() -> None:
    if True:  # flip to True while iterating to see the expected type errors
        # Problem provides only FunctionCost.
        prob_f = BenchmarkProblem(_OnlyFunction())

        # E: Should be rejected: algorithm/metric requires GradientCost but problem provides only FunctionCost.
        benchmark(
            problem=prob_f,
            algorithms=[
                AlgoNeedsGradient(),
                AlgoNeedsFunction(),
                AlgoNeedsFunctionGradient(),
            ],
            plot_metrics=[PlotNeedsCost(), PlotNeedsGradient()],
            table_metrics=[TableNeedsFunction(), TableNeedsGradient()],
        )


def _variance_sanity() -> None:
    # If contravariance is working, these assignments should type-check.
    # - An Algorithm[FunctionCost] is usable where Algorithm[FunctionGradientCost] is expected.
    alg_ok: Algorithm[FunctionGradientCost] = AlgoNeedsFunction()  # OK
    _ = alg_ok

    if True:  # flip to True while iterating to see the expected type errors
        # E: The reverse should be rejected.
        alg_bad: Algorithm[FunctionCost] = AlgoNeedsFunctionGradient()
        _ = alg_bad
