from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from decent_bench.utils.types import SupportedFrameworks

from ._abstracts._backend import _Backend


@dataclass(frozen=True, slots=True)
class _BackendSpec:
    cls: type[_Backend]
    init_kwargs: dict[str, Any]


_BACKEND_SPECS: dict[SupportedFrameworks, _BackendSpec] = {}
_BACKEND_INSTANCES: dict[SupportedFrameworks, _Backend] = {}
_ACTIVE_BACKEND_NAME: ContextVar[SupportedFrameworks | None] = ContextVar(
    "decent_bench.iop2.active_backend_name", default=None
)


def set_backend(backend: SupportedFrameworks) -> None:
    """
    Set the active backend for the current execution context.

    The first call sets the backend. Subsequent calls must use the same backend name;
    otherwise an error is raised.

    Args:
        backend: Backend name, SupportedFrameworks enum.

    Raises:
        KeyError: If the desired backend is not registered.
        RuntimeError: If a different backend was already set in this context.

    """
    current = _ACTIVE_BACKEND_NAME.get()
    if current is None:
        if not _is_backend_registered(backend):
            raise KeyError(f"Backend '{backend}' is not registered. Registered: {_backend_names()}")
        _ACTIVE_BACKEND_NAME.set(backend)
        return

    if current != backend:
        raise RuntimeError(f"Backend already set to '{current}', cannot set to '{backend}'.")


# Decorator for registering backends
def register_backend(
    backend: SupportedFrameworks,
    init_kwargs: dict[str, Any] | None = None,
) -> Callable[[type[_Backend]], type[_Backend]]:
    """
    Register a backend class under a backend.

    Backends are instantiated lazily upon first use (e.g. via `set_backend`).
    """

    def decorator(cls: type[_Backend]) -> type[_Backend]:
        if not issubclass(cls, _Backend):
            raise TypeError(f"Registered backend must be a subclass of _Backend, got {cls}")
        _BACKEND_SPECS[backend] = _BackendSpec(cls=cls, init_kwargs=dict(init_kwargs or {}))
        return cls

    return decorator


def _get_backend(backend: SupportedFrameworks | None = None) -> _Backend:
    """
    Return a cached backend instance, instantiating it lazily if needed.

    Args:
        backend: Optional backend name. If None, uses the currently active backend set by `set_backend`.

    Raises:
        RuntimeError: If no backend is set or instantiation of the backend fails.
        KeyError: If the specified backend is not registered.

    """
    if backend is None:
        backend = _ACTIVE_BACKEND_NAME.get()
        if backend is None:
            raise RuntimeError(
                "No backend has been set. Call set_backend(...) first, or instantiate a cost function "
                "to automatically set the backend based on the cost's framework."
            )
    elif not _is_backend_registered(backend):
        raise KeyError(f"Backend '{backend}' is not registered. Registered: {_backend_names()}")

    if backend in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[backend]

    spec = _get_backend_spec(backend)

    try:
        instance = spec.cls(**spec.init_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate backend '{backend}': {exc}") from exc

    _BACKEND_INSTANCES[backend] = instance
    return instance


def _backend_names() -> tuple[SupportedFrameworks, ...]:
    """Return the registered backend names."""
    return tuple(sorted(_BACKEND_SPECS.keys(), key=lambda x: x.value))


def _get_backend_spec(backend: SupportedFrameworks) -> _BackendSpec:
    try:
        return _BACKEND_SPECS[backend]
    except KeyError as exc:
        raise KeyError(f"Backend '{backend}' is not registered. Registered: {_backend_names()}") from exc


def _is_backend_registered(backend: SupportedFrameworks) -> bool:
    return backend in _BACKEND_SPECS


# PyTest utility functions for clearing backend state between tests


def _clear_backend_instances() -> None:
    """
    Clear all cached backend instances.

    This is mainly useful for tests.
    """
    _BACKEND_INSTANCES.clear()


def _clear_backend_registry() -> None:
    """
    Clear the backend registry.

    This is mainly useful for tests.
    """
    _BACKEND_SPECS.clear()
    _BACKEND_INSTANCES.clear()


def _reset_backend() -> None:
    """
    Reset the active backend for the current context.

    Intended mainly for tests or tightly scoped execution.
    """
    _ACTIVE_BACKEND_NAME.set(None)
