from dataclasses import dataclass
from functools import reduce
from operator import add

import networkx as nx

import decent_bench.centralized_algorithms as ca
import decent_bench.metrics.metric_utils as utils
import decent_bench.utils.interoperability as iop
from decent_bench import benchmark, benchmark_problem, costs
from decent_bench.costs import LinearRegressionCost
from decent_bench.datasets import SyntheticClassificationData
from decent_bench.distributed_algorithms import (
    ADMM,
    ATC,
    ATG,
    DGD,
    DLM,
    ED,
    EXTRA,
    NIDS,
    Algorithm,
    ATCTracking,
    AugDGM,
    SimpleGT,
    WangElia,
)
from decent_bench.metrics.plot_metrics import (
    DEFAULT_PLOT_METRICS,
    GradientNormPerIteration,
    RegretPerIteration,
)
from decent_bench.metrics.table_metrics import (
    DEFAULT_TABLE_METRICS,
    GradientNorm,
    Regret,
)
from decent_bench.networks import P2PNetwork
from decent_bench.schemes import *
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


@dataclass(eq=False)
class TestAlg(Algorithm[costs.HessianCost]):
    r"""
    Distributed gradient descent characterized by the update step below.

    .. math::
        \mathbf{x}_{i, k+1} = (\sum_{j} \mathbf{W}_{ij} \mathbf{x}_{j,k}) - \rho \nabla f_i(\mathbf{x}_{i,k})

    where
    :math:`\mathbf{x}_{i, k}` is agent i's local optimization variable at iteration k,
    j is a neighbor of i or i itself,
    :math:`\mathbf{W}_{ij}` is the metropolis weight between agent i and j,
    :math:`\rho` is the step size,
    and :math:`f_i` is agent i's local cost function.

    """

    step_size: float
    iterations: int = 100  # pyright: ignore[reportIncompatibleMethodOverride]
    name: str = "DGD"  # pyright: ignore[reportIncompatibleMethodOverride]

    type NetworkType = P2PNetwork[costs.HessianCost]

    def initialize(self, network: NetworkType) -> None:  # noqa: D102
        pass

    def step(self, network: NetworkType, iteration: int) -> None:  # noqa: D102
        pass


class TestCost(costs.GradientCost):
    r"""
    Linear regression cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2

    or in the general quadratic form

    .. math::
        f(\mathbf{x})
        = \frac{1}{2} \mathbf{x}^T\mathbf{A}^T\mathbf{Ax}
        - (\mathbf{A}^T \mathbf{b})^T \mathbf{x}
        + \frac{1}{2} \mathbf{b}^T\mathbf{b}
    """

    def __init__(self, A: Array, b: Array):  # noqa: N803
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(
                f"Dimension mismatch: A has {iop.shape(A)[0]} rows but b has {iop.shape(b)[0]} elements"
            )
        self.inner = costs.QuadraticCost(
            iop.dot(iop.transpose(A), A),
            -iop.dot(iop.transpose(A), b),
            float(0.5 * iop.dot(b, b)),
        )
        self.A = A
        self.b = b

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.inner.shape

    @property
    def framework(self) -> SupportedFrameworks:  # noqa: D102
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:  # noqa: D102
        return SupportedDevices.CPU

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        return self.inner.m_smooth

    @property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \begin{array}{ll}
                \min_i \lambda_i, & \text{if } \min_i \lambda_i > 0, \\
                0, & \text{if } \min_i \lambda_i = 0, \\
                \text{NaN}, & \text{if } \min_i \lambda_i < 0
            \end{array}

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return self.inner.m_cvx

    def gradient(self, x: Array) -> Array:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.inner.gradient(x)

    def __add__(self, other: costs.Cost) -> costs.Cost:
        """Add another cost function."""
        return self.inner + other


def create_test() -> benchmark_problem.BenchmarkProblem[TestCost]:  # noqa: D103
    network_structure = nx.random_regular_graph(3, 50, seed=0)
    dataset = SyntheticClassificationData(
        n_classes=2,
        n_partitions=50,
        n_samples_per_partition=10,
        n_features=3,
        framework=SupportedFrameworks.NUMPY,
        device=SupportedDevices.CPU,
        seed=0,
    )
    costs_list = [TestCost(*p) for p in dataset.training_partitions()]
    sum_cost = reduce(add, costs_list)
    x_optimal = ca.accelerated_gradient_descent(
        sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16
    )
    agent_activations = [AlwaysActive()] * 50
    message_compression = NoCompression()
    message_noise = NoNoise()
    message_drop = NoDrops()

    return benchmark_problem.BenchmarkProblem(
        network_structure=network_structure,
        costs=costs_list,
        x_optimal=x_optimal,
        agent_activations=agent_activations,
        message_compression=message_compression,
        message_noise=message_noise,
        message_drop=message_drop,
        agent_state_snapshot_period=1,
    )


if __name__ == "__main__":
    if True:  # flip to True while iterating to see the expected type errors
        # Case 1: Type checker catches this error: TestAlg requires HessianCost, problem provides GradientCost
        problem = create_test()
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ATC(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                SimpleGT(iterations=1000, step_size=0.001),
                TestAlg(iterations=1000, step_size=0.001),  # TYPE ERROR CAUGHT!
            ],
            benchmark_problem=problem,
            n_trials=10,
        )

    if True:  # flip to False while iterating to skip to the passing cases
        # Case 2: Type checker catches this error: ADMM requires ProximalCost, problem provides GradientCost
        problem2 = create_test()
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ATC(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                SimpleGT(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),  # TYPE ERROR CAUGHT!
            ],
            benchmark_problem=problem2,
            n_trials=10,
        )

    if True:  # flip to True while iterating to see the expected type errors
        # Case 3: Type checker catches this error: TestAlg requires hessian problem provides GradientCost
        problem3 = create_test()
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ATC(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                SimpleGT(iterations=1000, step_size=0.001),
                TestAlg(iterations=1000, step_size=0.001),  # TYPE ERROR CAUGHT!
            ],
            benchmark_problem=problem3,
            n_trials=10,
        )

    if True:  # flip to False while iterating to skip to the passing cases
        # Case 4: Type checker catches this error: ADMM requires ProximalCost and TestAlg requires hessian
        # Problem provides GradientCost
        problem4 = create_test()
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ATC(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                SimpleGT(iterations=1000, step_size=0.001),
                ADMM(iterations=1000, rho=10, alpha=0.3),  # TYPE ERROR CAUGHT!
                TestAlg(iterations=1000, step_size=0.001),  # TYPE ERROR CAUGHT!
            ],
            benchmark_problem=problem4,
            n_trials=10,
        )

    if True:  # flip to True while iterating to see the expected type errors
        # Case 5: Should fail (if metrics are typechecked), problem provides GradientCost
        # but Regret requires FunctionCost
        problem5 = create_test()
        benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.001),
                ATC(iterations=1000, step_size=0.001),
                ED(iterations=1000, step_size=0.001),
                SimpleGT(iterations=1000, step_size=0.001),
            ],
            benchmark_problem=problem5,
            n_trials=10,
            plot_metrics=[GradientNormPerIteration(), RegretPerIteration()],
            table_metrics=[GradientNorm([utils.single]), Regret([utils.single])],
        )

    # Case 6: Should pass, problem provides GradientCost
    problem6 = create_test()
    benchmark.benchmark(
        algorithms=[
            DGD(iterations=1000, step_size=0.001),
            ATC(iterations=1000, step_size=0.001),
            ED(iterations=1000, step_size=0.001),
            SimpleGT(iterations=1000, step_size=0.001),
        ],
        benchmark_problem=problem6,
        n_trials=10,
        plot_metrics=[GradientNormPerIteration()],
        table_metrics=[GradientNorm([utils.single])],
    )

    # Case 7: Should pass, problem provides ProximalCost and GradientCost
    problem7 = benchmark_problem.create_regression_problem(LinearRegressionCost)
    benchmark.benchmark(
        benchmark_problem=problem7,
        algorithms=[
            DGD(iterations=1000, step_size=0.001),
            ATC(iterations=1000, step_size=0.001),
            ED(iterations=1000, step_size=0.001),
            SimpleGT(iterations=1000, step_size=0.001),
            AugDGM(iterations=1000, step_size=0.001),
            WangElia(iterations=1000, step_size=0.001),
            EXTRA(iterations=1000, step_size=0.001),
            ATCTracking(iterations=1000, step_size=0.001),
            NIDS(iterations=1000, step_size=0.001),
            ADMM(iterations=1000, rho=10, alpha=0.3),
            ATG(iterations=1000, rho=0.9, alpha=0.3),
            DLM(iterations=1000, step_size=0.001, penalty=0.1),
        ],
        n_trials=10,
    )

    # Case 8: Should pass, problem provides GradientCost
    problem8 = benchmark_problem.create_regression_problem(LinearRegressionCost)
    benchmark.benchmark(
        benchmark_problem=problem8,
        algorithms=[
            DGD(iterations=1000, step_size=0.001),
            ATC(iterations=1000, step_size=0.001),
            ED(iterations=1000, step_size=0.001),
            SimpleGT(iterations=1000, step_size=0.001),
        ],
        n_trials=10,
    )
