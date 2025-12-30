import pytest

from decent_bench.costs import (
    Cost,
    FunctionCost,
    FunctionGradientCost,
    FunctionGradientHessianCost,
    FunctionGradientHessianProximalCost,
    FunctionGradientProximalCost,
    FunctionHessianCost,
    FunctionHessianProximalCost,
    FunctionProximalCost,
    GradientCost,
    GradientHessianCost,
    GradientHessianProximalCost,
    GradientProximalCost,
    HessianCost,
    HessianProximalCost,
    ProximalCost,
)


class DummyFunctionGradientHessianProximal(FunctionGradientHessianProximalCost):
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def gradient(self, x):
        pass

    def hessian(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyFunctionGradientHessian(FunctionGradientHessianCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def gradient(self, x):
        pass

    def hessian(self, x):
        pass


class DummyFunctionGradientProximal(FunctionGradientProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def gradient(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyFunctionHessianProximal(FunctionHessianProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def hessian(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyGradientHessianProximal(GradientHessianProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def gradient(self, x):
        pass

    def hessian(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyFunctionGradient(FunctionGradientCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def gradient(self, x):
        pass


class DummyFunctionHessian(FunctionHessianCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def hessian(self, x):
        pass


class DummyFunctionProximal(FunctionProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyGradientHessian(GradientHessianCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def gradient(self, x):
        pass

    def hessian(self, x):
        pass


class DummyGradientProximal(GradientProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def gradient(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyHessianProximal(HessianProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def hessian(self, x):
        pass

    def proximal(self, x, rho):
        pass


class DummyFunctionCost(FunctionCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def function(self, x):
        pass


class DummyGradientCost(GradientCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def gradient(self, x):
        pass


class DummyHessianCost(HessianCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def hessian(self, x):
        pass


class DummyProximalCost(ProximalCost):  # noqa: D101
    def device(self):
        pass

    def framework(self):
        pass

    def m_cvx(self):
        pass

    def m_smooth(self):
        pass

    def shape(self):
        pass

    def __add__(self, other):
        pass

    def proximal(self, x, rho):
        pass


def _create_dummy_algorithm(cost_type: type[Cost]):
    """Create a dummy algorithm that requires the specified cost type."""
    from decent_bench.distributed_algorithms import Algorithm

    class _DummyAlgorithm(Algorithm[cost_type]):
        @property
        def iterations(self) -> int:
            return 1

        @property
        def name(self) -> str:
            return "DummyAlgorithm"

        def initialize(self, network) -> None:
            pass

        def step(self, network, iteration: int) -> None:
            pass

    return _DummyAlgorithm()


def _create_dummy_plot_metric(cost_type: type[Cost]):
    """Create a dummy plot metric that requires the specified cost type."""
    from decent_bench.metrics.plot_metrics import PlotMetric

    class _DummyPlotMetric(PlotMetric[cost_type]):
        @property
        def plot_description(self) -> str:
            return "dummy"

        def get_data_from_trial(self, agents, problem):
            return []

    return _DummyPlotMetric()


def _create_dummy_table_metric(cost_type: type[Cost]):
    """Create a dummy table metric that requires the specified cost type."""
    from decent_bench.metrics.table_metrics import TableMetric

    class _DummyTableMetric(TableMetric[cost_type]):
        def __init__(self):
            super().__init__(statistics=[])

        @property
        def table_description(self) -> str:
            return "dummy"

        def get_data_from_trial(self, agents, problem):
            return []

    return _DummyTableMetric()


def _create_dummy_benchmark_problem(costs):
    """Create a dummy benchmark problem with the specified costs."""
    from decent_bench.benchmark import BenchmarkProblem

    return BenchmarkProblem(None, None, costs, 1, None, None, None, None)


# All available cost types
ALL_COST_CLASSES = [
    (FunctionCost, DummyFunctionCost),
    (GradientCost, DummyGradientCost),
    (HessianCost, DummyHessianCost),
    (ProximalCost, DummyProximalCost),
    (FunctionGradientCost, DummyFunctionGradient),
    (FunctionHessianCost, DummyFunctionHessian),
    (FunctionProximalCost, DummyFunctionProximal),
    (GradientHessianCost, DummyGradientHessian),
    (GradientProximalCost, DummyGradientProximal),
    (HessianProximalCost, DummyHessianProximal),
    (FunctionGradientHessianCost, DummyFunctionGradientHessian),
    (FunctionGradientProximalCost, DummyFunctionGradientProximal),
    (FunctionHessianProximalCost, DummyFunctionHessianProximal),
    (GradientHessianProximalCost, DummyGradientHessianProximal),
    (FunctionGradientHessianProximalCost, DummyFunctionGradientHessianProximal),
]


def _assert_inheritance(cost: Cost, primary_type: type[Cost], expected_parents: list[type[Cost]]) -> None:
    """Helper to assert that a cost inherits from all expected parent types and ONLY those types."""
    name = cost.__class__.__name__
    primary_name = primary_type.__name__

    # Check that all expected parents are present
    for parent_type in expected_parents:
        parent_name = parent_type.__name__
        assert isinstance(cost, parent_type), f"{name} inherits from {primary_name} but not {parent_name}"

    # Check that no unexpected parents are present
    # Get all possible cost types from the inheritance map
    all_cost_types = {x for x, _ in ALL_COST_CLASSES}
    # The expected types include the primary type and all expected parents
    expected_types = set(expected_parents) | {primary_type}
    # Unexpected types are all types not in the expected set
    unexpected_types = all_cost_types - expected_types

    for unexpected_type in unexpected_types:
        unexpected_name = unexpected_type.__name__
        assert not isinstance(cost, unexpected_type), (
            f"{name} inherits from {primary_name} and should NOT inherit from {unexpected_name}, but it does"
        )


# Define the inheritance hierarchy for each cost type
COST_INHERITANCE_MAP = {
    FunctionGradientHessianProximalCost: [
        FunctionCost,
        GradientCost,
        HessianCost,
        ProximalCost,
        FunctionGradientCost,
        FunctionHessianCost,
        FunctionProximalCost,
        GradientHessianCost,
        GradientProximalCost,
        HessianProximalCost,
        FunctionGradientHessianCost,
        FunctionGradientProximalCost,
        FunctionHessianProximalCost,
        GradientHessianProximalCost,
        Cost,
    ],
    FunctionGradientHessianCost: [
        FunctionCost,
        GradientCost,
        HessianCost,
        FunctionGradientCost,
        FunctionHessianCost,
        GradientHessianCost,
        Cost,
    ],
    FunctionGradientProximalCost: [
        FunctionCost,
        GradientCost,
        ProximalCost,
        FunctionGradientCost,
        FunctionProximalCost,
        GradientProximalCost,
        Cost,
    ],
    FunctionHessianProximalCost: [
        FunctionCost,
        HessianCost,
        ProximalCost,
        FunctionHessianCost,
        FunctionProximalCost,
        HessianProximalCost,
        Cost,
    ],
    GradientHessianProximalCost: [
        GradientCost,
        HessianCost,
        ProximalCost,
        GradientHessianCost,
        GradientProximalCost,
        HessianProximalCost,
        Cost,
    ],
    FunctionGradientCost: [FunctionCost, GradientCost, Cost],
    FunctionHessianCost: [FunctionCost, HessianCost, Cost],
    FunctionProximalCost: [FunctionCost, ProximalCost, Cost],
    GradientHessianCost: [GradientCost, HessianCost, Cost],
    GradientProximalCost: [GradientCost, ProximalCost, Cost],
    HessianProximalCost: [HessianCost, ProximalCost, Cost],
    FunctionCost: [Cost],
    GradientCost: [Cost],
    HessianCost: [Cost],
    ProximalCost: [Cost],
}


@pytest.mark.parametrize(
    "cost",
    [cost() for _, cost in ALL_COST_CLASSES],
)
def test_inheritance_consistency(cost: Cost) -> None:
    """Test that cost functions correctly inherit from their declared superclasses."""
    # Find the most specific type that matches this cost
    for cost_type, expected_parents in COST_INHERITANCE_MAP.items():
        if isinstance(cost, cost_type):
            _assert_inheritance(cost, cost_type, expected_parents)
            return

    raise AssertionError(f"Unknown cost class: {cost.__class__.__name__}")


# Tests for _ensure_benchmark_compatibility


@pytest.mark.parametrize(
    "required_cost_type,provided_cost_class",
    [
        (required_type, provided_class)
        for required_type, _ in ALL_COST_CLASSES
        for _, provided_class in ALL_COST_CLASSES
    ],
)
def test_ensure_benchmark_compatibility_algorithm_all_combinations(
    required_cost_type: type[Cost], provided_cost_class: type[Cost]
) -> None:
    """Test algorithm compatibility for all cost type combinations."""
    from decent_bench.benchmark import _ensure_benchmark_compatibility

    provided_cost = provided_cost_class()
    benchmark_problem = _create_dummy_benchmark_problem([provided_cost])
    algorithm = _create_dummy_algorithm(required_cost_type)

    # Check if the provided cost is an instance of the required type
    is_compatible = isinstance(provided_cost, required_cost_type)

    if is_compatible:
        # Should not raise an exception
        _ensure_benchmark_compatibility(benchmark_problem, [algorithm], None, None)
    else:
        # Should raise TypeError
        with pytest.raises(TypeError, match="Incompatible algorithm"):
            _ensure_benchmark_compatibility(benchmark_problem, [algorithm], None, None)


@pytest.mark.parametrize(
    "required_cost_type,provided_cost_class",
    [
        (required_type, provided_class)
        for required_type, _ in ALL_COST_CLASSES
        for _, provided_class in ALL_COST_CLASSES
    ],
)
def test_ensure_benchmark_compatibility_plot_metric_all_combinations(
    required_cost_type: type[Cost], provided_cost_class: type[Cost]
) -> None:
    """Test plot metric compatibility for all cost type combinations."""
    from decent_bench.benchmark import _ensure_benchmark_compatibility

    provided_cost = provided_cost_class()
    benchmark_problem = _create_dummy_benchmark_problem([provided_cost])
    plot_metric = _create_dummy_plot_metric(required_cost_type)

    # Check if the provided cost is an instance of the required type
    is_compatible = isinstance(provided_cost, required_cost_type)

    if is_compatible:
        # Should not raise an exception
        _ensure_benchmark_compatibility(benchmark_problem, [], [plot_metric], None)
    else:
        # Should raise TypeError
        with pytest.raises(TypeError, match="Incompatible plot metric"):
            _ensure_benchmark_compatibility(benchmark_problem, [], [plot_metric], None)


@pytest.mark.parametrize(
    "required_cost_type,provided_cost_class",
    [
        (required_type, provided_class)
        for required_type, _ in ALL_COST_CLASSES
        for _, provided_class in ALL_COST_CLASSES
    ],
)
def test_ensure_benchmark_compatibility_table_metric_all_combinations(
    required_cost_type: type[Cost], provided_cost_class: type[Cost]
) -> None:
    """Test table metric compatibility for all cost type combinations."""
    from decent_bench.benchmark import _ensure_benchmark_compatibility

    provided_cost = provided_cost_class()
    benchmark_problem = _create_dummy_benchmark_problem([provided_cost])
    table_metric = _create_dummy_table_metric(required_cost_type)

    # Check if the provided cost is an instance of the required type
    is_compatible = isinstance(provided_cost, required_cost_type)

    if is_compatible:
        # Should not raise an exception
        _ensure_benchmark_compatibility(benchmark_problem, [], None, [table_metric])
    else:
        # Should raise TypeError
        with pytest.raises(TypeError, match="Incompatible table metric"):
            _ensure_benchmark_compatibility(benchmark_problem, [], None, [table_metric])


def test_ensure_benchmark_compatibility_with_empty_costs():
    """Test that empty costs list doesn't raise errors."""
    from decent_bench.benchmark import _ensure_benchmark_compatibility

    benchmark_problem = _create_dummy_benchmark_problem([])

    alg = _create_dummy_algorithm(GradientCost)
    plot_metric = _create_dummy_plot_metric(GradientCost)
    table_metric = _create_dummy_table_metric(GradientCost)

    # This should not raise any exception since there are no costs to check
    _ensure_benchmark_compatibility(benchmark_problem, [alg], [plot_metric], [table_metric])


def test_ensure_benchmark_compatibility_with_none_metrics():
    """Test that None values for metrics are handled correctly."""
    from decent_bench.benchmark import _ensure_benchmark_compatibility

    cost = DummyFunctionGradient()
    benchmark_problem = _create_dummy_benchmark_problem([cost])
    alg = _create_dummy_algorithm(GradientCost)

    # This should not raise any exception
    _ensure_benchmark_compatibility(benchmark_problem, [alg], None, None)
