"""
Lightweight wrapper around backend-native arrays.

The :class:`Array` class wraps a single value of the active backend's framework type.
Under the single-active-backend invariant maintained by
:mod:`decent_bench.utils.interoperability_2`, every :class:`Array` at runtime holds a
value from the same framework, so operators can dispatch directly to the active backend
without per-call isinstance dispatch.

Operator contract is *strict*: binary arithmetic and indexing accept either another
:class:`Array` or a Python scalar (``int``/``float``). Pass other framework-native
arrays through :func:`decent_bench.utils.interoperability_2.get_item` and friends, not
through the operator path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NoReturn, Self

if TYPE_CHECKING:
    from decent_bench.utils.interoperability_2._abstracts._backend import _Backend
    from decent_bench.utils.types import ArrayKey, SupportedArrayTypes


class _NoBackendSet:
    """Sentinel that raises a helpful error on any attribute access."""

    __slots__ = ()

    def __getattr__(self, name: str) -> NoReturn:
        raise RuntimeError(
            "Array operations require an active backend. Call "
            "decent_bench.utils.interoperability_2.set_backend(...) first."
        )


# Module-level cache for the active backend. Bound by
# :func:`decent_bench.utils.interoperability_2.set_backend`. Reading ``_BACKEND`` on
# the hot path is a single global-name load — no ContextVar lookup, no get_backend()
# function call.
_BACKEND: _Backend = _NoBackendSet()  # type: ignore[assignment]


def _set_active_backend(backend: _Backend) -> None:
    """
    Bind the active backend.

    Called from :func:`decent_bench.utils.interoperability_2.set_backend`. Not part of
    the public API.
    """
    global _BACKEND  # noqa: PLW0603
    _BACKEND = backend


class Array:  # noqa: PLR0904
    """
    Wrapper around a single backend-native array.

    Storage is a single attribute (``value``) declared via ``__slots__``; instances
    have no ``__dict__``. Every operator delegates to the active backend resolved via
    the module-level ``_BACKEND`` reference, so dispatch is one attribute load plus the
    backend method call.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:  # noqa: ANN401
        """
        Wrap ``value`` in an :class:`Array`.

        Args:
            value: A backend-native array (or scalar) to wrap. The attribute is typed
                as :class:`typing.Any` because the wrapper is intentionally type-erased
                — backend code accesses :attr:`value` knowing the framework type, and
                typing it more strictly forces a ``cast`` at every call site without
                runtime benefit.

        """
        self.value: Any = value

    # Binary arithmetic ----------------------------------------------------

    def __add__(self, other: Array | int | float) -> Array:
        return _BACKEND.add(self, other)  # type: ignore[arg-type]

    def __radd__(self, other: int | float) -> Array:
        return _BACKEND.add(other, self)  # type: ignore[arg-type]

    def __sub__(self, other: Array | int | float) -> Array:
        return _BACKEND.sub(self, other)  # type: ignore[arg-type]

    def __rsub__(self, other: int | float) -> Array:
        return _BACKEND.sub(other, self)  # type: ignore[arg-type]

    def __mul__(self, other: Array | int | float) -> Array:
        return _BACKEND.mul(self, other)  # type: ignore[arg-type]

    def __rmul__(self, other: int | float) -> Array:
        return _BACKEND.mul(other, self)  # type: ignore[arg-type]

    def __truediv__(self, other: Array | int | float) -> Array:
        return _BACKEND.div(self, other)  # type: ignore[arg-type]

    def __rtruediv__(self, other: int | float) -> Array:
        return _BACKEND.div(other, self)  # type: ignore[arg-type]

    def __matmul__(self, other: Array) -> Array:
        return _BACKEND.matmul(self, other)

    def __rmatmul__(self, other: Array) -> Array:
        return _BACKEND.matmul(other, self)

    def __pow__(self, other: float) -> Array:
        return _BACKEND.pow(self, other)

    # In-place arithmetic --------------------------------------------------
    #
    # The backend handles the framework's mutability semantics: numpy/pytorch mutate
    # `value` in place, jax/tensorflow rebind it. In every case the returned object is
    # the same wrapper instance, so we just discard the return and yield ``self``.

    def __iadd__(self, other: Array | int | float) -> Self:
        _BACKEND.iadd(self, other)  # type: ignore[arg-type]
        return self

    def __isub__(self, other: Array | int | float) -> Self:
        _BACKEND.isub(self, other)  # type: ignore[arg-type]
        return self

    def __imul__(self, other: Array | int | float) -> Self:
        _BACKEND.imul(self, other)  # type: ignore[arg-type]
        return self

    def __itruediv__(self, other: Array | int | float) -> Self:
        _BACKEND.idiv(self, other)  # type: ignore[arg-type]
        return self

    def __ipow__(self, other: float) -> Self:
        _BACKEND.ipow(self, other)
        return self

    # Unary ----------------------------------------------------------------

    def __neg__(self) -> Array:
        return _BACKEND.negative(self)

    def __abs__(self) -> Array:
        return _BACKEND.absolute(self)

    # Indexing -------------------------------------------------------------

    def __getitem__(self, key: ArrayKey) -> Array:
        return _BACKEND.get_item(self, key)

    def __setitem__(self, key: ArrayKey, value: Array | int | float) -> None:
        if not isinstance(value, Array):
            value = Array(value)
        _BACKEND.set_item(self, key, value)

    # Containers / iteration ----------------------------------------------

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self) -> Iterator[SupportedArrayTypes]:
        return iter(self.value)

    # Coercion -------------------------------------------------------------

    def __array__(self) -> SupportedArrayTypes:  # noqa: PLW3201
        return self.value

    def __float__(self) -> float:
        return float(_BACKEND.astype(self, float))

    # Repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Array({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)
