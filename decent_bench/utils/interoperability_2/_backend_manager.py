from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from contextvars import ContextVar

from decent_bench.utils.types import SupportedFrameworks

from ._abstracts._backend import _Backend

_BACKEND_REGISTRY: dict[SupportedFrameworks, type[_Backend]] = {}
_BACKEND_INSTANCES: dict[SupportedFrameworks, _Backend] = {}
_BACKEND_ALIASES: dict[str, SupportedFrameworks] = {}
_ACTIVE_BACKEND: ContextVar[SupportedFrameworks | None] = ContextVar("decent_bench.iop2.active_backend", default=None)


def set_backend(backend: SupportedFrameworks | str) -> None:
    """
    Set the active backend for the current execution context.

    The first call binds the backend; subsequent calls must use the same backend or a
    :class:`RuntimeError` is raised. This single-backend invariant lets the rest of the
    interoperability layer skip framework dispatch and isinstance checks.

    Backend modules are auto-imported on demand: the first call to ``set_backend("pytorch")``
    triggers import of ``decent_bench.utils.interoperability_2._pytorch``, whose ``__init__``
    is expected to register the backend via :func:`register_backend`.

    Args:
        backend: A :class:`SupportedFrameworks` value, its canonical string (e.g.
            ``"numpy"``, ``"pytorch"``), or any alias declared by the backend at
            registration time. Aliases are only resolvable after the backend module has
            been loaded; pass the canonical name on the first call to trigger autoload.

    Raises:
        ImportError: If the backend module cannot be imported (e.g. due to a missing dependency).
        RuntimeError: If a different backend is already active in this context.

    """
    requested = _normalize(backend)

    if requested not in _BACKEND_REGISTRY:
        try:
            _auto_import(requested)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import the backend module for '{requested.value}'. Ensure the "
                "corresponding backend package is installed and importable."
            ) from exc

    current = _ACTIVE_BACKEND.get()
    if current is not None and current != requested:
        raise RuntimeError(
            f"Backend already set to '{current.value}', cannot set to '{requested.value}'. "
            "A single execution context may only use one backend."
        )

    if current is None:
        _ACTIVE_BACKEND.set(requested)


def get_backend() -> _Backend:
    """
    Return the active backend instance.

    Raises:
        RuntimeError: If no backend has been set in this context.

    """
    active = _ACTIVE_BACKEND.get()
    if active is None:
        raise RuntimeError(
            "No backend has been set. Call set_backend(...) first, or instantiate a cost "
            "function to automatically set the backend based on the cost's framework."
        )
    return _instantiate(active)


def register_backend(
    backend: SupportedFrameworks,
    aliases: Iterable[str] | None = None,
) -> Callable[[type[_Backend]], type[_Backend]]:
    """
    Register a backend class under a :class:`SupportedFrameworks` value.

    Backends are instantiated lazily on first use. Re-registering replaces the previous
    class and discards any cached instance, but keeps existing aliases (which still
    point to the same canonical name).

    Args:
        backend: Canonical backend identifier.
        aliases: Optional extra names users may pass to :func:`set_backend`. The
            canonical string (``backend.value``) is always accepted and need not be
            listed here. An alias that collides with a canonical name or another
            backend's alias raises :class:`ValueError` when the decorator is applied.

    """

    def decorator(cls: type[_Backend]) -> type[_Backend]:
        if not issubclass(cls, _Backend):
            raise TypeError(f"Registered backend must be a subclass of _Backend, got {cls}")
        _BACKEND_REGISTRY[backend] = cls
        _BACKEND_INSTANCES.pop(backend, None)
        for alias in aliases or ():
            _register_alias(alias, backend)
        return cls

    return decorator


def reset_backend() -> None:
    """
    Clear the active backend and all cached instances for the current context.

    Intended for tests or tightly scoped execution; not part of normal use. Registry
    entries (classes and aliases) are preserved.
    """
    _ACTIVE_BACKEND.set(None)
    _BACKEND_INSTANCES.clear()


def _register_alias(alias: str, backend: SupportedFrameworks) -> None:
    if alias in {f.value for f in SupportedFrameworks} and SupportedFrameworks(alias) != backend:
        raise ValueError(f"Alias '{alias}' collides with the canonical name of '{SupportedFrameworks(alias).value}'.")
    existing = _BACKEND_ALIASES.get(alias)
    if existing is not None and existing != backend:
        raise ValueError(f"Alias '{alias}' is already registered for backend '{existing.value}'.")
    _BACKEND_ALIASES[alias] = backend


def _normalize(backend: SupportedFrameworks | str) -> SupportedFrameworks:
    if isinstance(backend, SupportedFrameworks):
        return backend
    if backend in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[backend]
    try:
        return SupportedFrameworks(backend)
    except ValueError as exc:
        valid = ", ".join(f.value for f in SupportedFrameworks)
        raise KeyError(f"Unknown backend '{backend}'. Valid backends: {valid}.") from exc


def _instantiate(backend: SupportedFrameworks) -> _Backend:
    if backend in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[backend]

    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise KeyError(
            f"Backend '{backend.value}' is not registered. Ensure the corresponding backend module is importable."
        )

    instance = cls()
    _BACKEND_INSTANCES[backend] = instance
    return instance


def _auto_import(backend: SupportedFrameworks) -> None:
    current_module = __name__.rsplit(".", 1)[0]
    module_name = current_module + f"._{backend.value}"
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import the backend module for '{backend.value}'. Ensure the "
            "corresponding backend package is installed and importable."
        ) from exc
