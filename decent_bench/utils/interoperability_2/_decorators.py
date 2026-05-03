"""
Decorator that bridges :class:`Cost` superclass signatures (typed in :class:`Array`) with
framework-native subclass implementations.

Single-backend semantics make this decorator dramatically simpler than the v1 version: no
framework dispatch, no cross-framework conversion, no ``to_array_like`` magic — just
unwrap input :class:`Array` values to their native form, call the subclass method, and
re-wrap the return if the superclass declared ``-> Array``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from decent_bench.utils.array import Array


def autodecorate_cost_method[T: Callable[..., Any]](superclass_method: T) -> Callable[[Callable[..., Any]], T]:
    """
    Decorate a Cost method override so its body can use raw framework arrays.

    Each :class:`Array` argument is unwrapped to its underlying value before the call.
    If the *superclass* method's return annotation is :class:`Array`, the return value
    is re-wrapped in :class:`Array` (unless already wrapped). All other arguments and
    return values pass through unchanged.

    Args:
        superclass_method: The base-class method being overridden (e.g. ``Cost.function``).
            Used solely to look up the declared return type at decoration time.

    Example:
        class LinearRegressionCost(EmpiricalRiskCost):
            @autodecorate_cost_method(EmpiricalRiskCost.gradient)
            def gradient(self, x: NDArray[float64], indices: ...) -> NDArray[float64]:
                # ``x`` arrives as a numpy ndarray; the wrapper unwraps the caller's Array.
                return self.A.T @ (self.A @ x - self.b) / self.n_samples
                # Return value is wrapped back into Array because EmpiricalRiskCost.gradient
                # is annotated ``-> Array``.

    """
    return_is_array = superclass_method.__annotations__.get("return") is Array

    def decorator(func: Callable[..., Any]) -> T:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            new_args = [a.value if isinstance(a, Array) else a for a in args]
            new_kwargs = {k: (v.value if isinstance(v, Array) else v) for k, v in kwargs.items()}
            result = func(self, *new_args, **new_kwargs)
            if return_is_array and not isinstance(result, Array):
                return Array(result)
            return result

        return cast("T", wrapper)

    return decorator
