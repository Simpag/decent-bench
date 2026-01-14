from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class Cost(ABC):
    """Used by agents to evaluate the cost and its derivatives at a certain x."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Required shape of x."""

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Alias for :attr:`shape`."""
        return self.shape

    @property
    @abstractmethod
    def framework(self) -> SupportedFrameworks:
        """
        The framework used by this cost function.

        Make sure that all :class:`decent_bench.utils.array.Array` objects returned by this cost function's methods
        use this framework.

        """

    @property
    @abstractmethod
    def device(self) -> SupportedDevices:
        """
        The device used by this cost function.

        Make sure that all :class:`decent_bench.utils.array.Array` objects returned by this cost function's methods
        use this device.

        """

    @property
    @abstractmethod
    def m_smooth(self) -> float:
        r"""
        Lipschitz constant of the cost function's gradient.

        The gradient's Lipschitz constant m_smooth is the smallest value such that
        :math:`\| \nabla f(\mathbf{x_1}) - \nabla f(\mathbf{x_2}) \| \leq \text{m_smooth}
        \cdot \|\mathbf{x_1} - \mathbf{x_2}\|`
        for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

        Returns:
            - non-negative finite number if function is L-smooth
            - ``np.inf`` if function is differentiable everywhere but not L-smooth
            - ``np.nan`` if function is not differentiable everywhere

        """

    @property
    @abstractmethod
    def m_cvx(self) -> float:
        r"""
        Convexity constant of the cost function.

        The convexity constant m_cvx is the largest value such that
        :math:`f(\mathbf{x_1}) \geq f(\mathbf{x_2})
        + \nabla f(\mathbf{x_2})^T (\mathbf{x_1} - \mathbf{x_2})
        + \frac{\text{m_cvx}}{2} \|\mathbf{x_1} - \mathbf{x_2}\|^2`
        for all :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

        Returns:
            - positive finite number if function is strongly convex
            - ``0`` if function is convex but not strongly convex
            - ``np.nan`` if function is not guaranteed to be convex

        """

    @abstractmethod
    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function to create a new one.

        :class:`~SumCost` can be used as the result of :meth:`__add__` by returning
        ``SumCost([self, other])``. However, it's often more efficient to preserve the cost function type if possible.
        For example, the addition of two :class:`~QuadraticCost` objects benefits from returning a new
        :class:`~QuadraticCost` instead of a :class:`~SumCost` as this preserves the closed
        form proximal solution and only requires one evaluation instead of two when calling
        :meth:`function <FunctionCost.function>`, :meth:`gradient <GradientCost.gradient>`
        and :meth:`hessian <HessianCost.hessian>`.
        """


class FunctionCost(Cost, ABC):
    """Cost function with function function method."""

    @abstractmethod
    def function(self, x: Array) -> float:
        """Evaluate function at x."""

    def evaluate(self, x: Array) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    def loss(self, x: Array) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)

    def f(self, x: Array) -> float:
        """Alias for :meth:`function`."""
        return self.function(x)


class GradientCost(Cost, ABC):
    """Cost function with gradient method."""

    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Gradient at x."""


class HessianCost(Cost, ABC):
    """Cost function with hessian method."""

    @abstractmethod
    def hessian(self, x: Array) -> Array:
        """Hessian at x."""


class ProximalCost(Cost, ABC):
    """Cost function with proximal method."""

    @abstractmethod
    def proximal(self, x: Array, rho: float) -> Array:
        r"""
        Proximal at x.

        The proximal operator is defined as:

        .. include:: snippets/proximal_operator.rst

        If the cost function's proximal does not have a closed form solution, it can be solved iteratively using
        :meth:`~decent_bench.centralized_algorithms.proximal_solver`.
        """


class FunctionGradientCost(FunctionCost, GradientCost, ABC):
    """Cost function with function and gradient methods."""


class FunctionHessianCost(FunctionCost, HessianCost, ABC):
    """Cost function with function and hessian methods."""


class FunctionProximalCost(FunctionCost, ProximalCost, ABC):
    """Cost function with function and proximal methods."""


class GradientHessianCost(GradientCost, HessianCost, ABC):
    """Cost function with gradient and hessian methods."""


class GradientProximalCost(GradientCost, ProximalCost, ABC):
    """Cost function with gradient and proximal methods."""


class HessianProximalCost(HessianCost, ProximalCost, ABC):
    """Cost function with hessian and proximal methods."""


class FunctionGradientHessianCost(FunctionGradientCost, FunctionHessianCost, GradientHessianCost, ABC):
    """Cost function with function, gradient and hessian methods."""


class FunctionGradientProximalCost(FunctionGradientCost, FunctionProximalCost, GradientProximalCost, ABC):
    """Cost function with function, gradient and proximal methods."""


class FunctionHessianProximalCost(FunctionHessianCost, FunctionProximalCost, HessianProximalCost, ABC):
    """Cost function with function, hessian and proximal methods."""


class GradientHessianProximalCost(GradientHessianCost, GradientProximalCost, HessianProximalCost, ABC):
    """Cost function with gradient, hessian and proximal methods."""


class FunctionGradientHessianProximalCost(
    FunctionGradientHessianCost,
    FunctionGradientProximalCost,
    FunctionHessianProximalCost,
    GradientHessianProximalCost,
    ABC,
):
    """Cost function with function, gradient, hessian and proximal methods."""
