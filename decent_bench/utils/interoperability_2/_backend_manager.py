from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from contextvars import ContextVar

from decent_bench.utils.array._array import _NoBackendSet, _set_active_backend
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._abstracts._backend import _Backend

_BACKEND_REGISTRY: dict[SupportedFrameworks, type[_Backend]] = {}
_BACKEND_INSTANCES: dict[SupportedFrameworks, _Backend] = {}
_BACKEND_ALIASES: dict[str, SupportedFrameworks] = {}
_ACTIVE_BACKEND: ContextVar[SupportedFrameworks | None] = ContextVar("decent_bench.iop2.active_backend", default=None)


def set_backend(
    backend: SupportedFrameworks | str,
    device: SupportedDevices | str = SupportedDevices.CPU,
) -> None:
    """
    Set the active backend (and target device) for the current execution context.

    The first call binds both the backend and the device; subsequent calls must use the
    same backend *and* the same device or a :class:`RuntimeError` is raised. This
    single-backend, single-device invariant lets the rest of the interoperability layer
    skip framework dispatch and isinstance checks, and lets backends construct array
    creation routines bound to a specific accelerator.

    Backend modules are auto-imported on demand: the first call to ``set_backend("pytorch")``
    triggers import of ``decent_bench.utils.interoperability_2._pytorch``, whose ``__init__``
    is expected to register the backend via :func:`register_backend`.

    Args:
        backend: A :class:`SupportedFrameworks` value, its canonical string (e.g.
            ``"numpy"``, ``"pytorch"``), or any alias declared by the backend at
            registration time. Aliases are only resolvable after the backend module has
            been loaded; pass the canonical name on the first call to trigger autoload.
        device: Target accelerator. Accepts a :class:`SupportedDevices` value or its
            string equivalent (``"cpu"``, ``"gpu"``, ``"mps"``). Defaults to CPU. The
            backend's array-creation methods produce arrays on this device by default.

    Note:
        Raises :class:`ImportError` if the backend module cannot be imported (e.g. due to
        a missing optional dependency); the failure originates from :func:`_auto_import`.

    Raises:
        RuntimeError: If a different backend (or the same backend with a different device)
            is already active in this context.

    """
    requested = _normalize(backend)
    requested_device = device if isinstance(device, SupportedDevices) else SupportedDevices(device)

    if requested not in _BACKEND_REGISTRY:
        _auto_import(requested)

    current = _ACTIVE_BACKEND.get()
    if current is not None and current != requested:
        raise RuntimeError(
            f"Backend already set to '{current.value}', cannot set to '{requested.value}'. "
            "A single execution context may only use one backend."
        )

    cached = _BACKEND_INSTANCES.get(requested)
    if cached is None:
        cls = _BACKEND_REGISTRY[requested]
        cached = cls(device=requested_device)
        _BACKEND_INSTANCES[requested] = cached
    elif cached.device != requested_device:
        raise RuntimeError(
            f"Backend '{requested.value}' already configured with device "
            f"'{cached.device.value}', cannot reconfigure to '{requested_device.value}'."
        )

    if current is None:
        _ACTIVE_BACKEND.set(requested)
        # Cache the backend on each module that has its own ``_BACKEND`` slot (Array,
        # iop free functions, iop RNG) so dispatch is a single global-name load
        # instead of a ContextVar + dict lookup per call.
        _notify_backend_subscribers(cached)


def _notify_backend_subscribers(backend: _Backend) -> None:
    """
    Push the active backend into every module that caches it as a module global.

    Iop modules are imported lazily here (rather than at top of file) to avoid the
    circular ``_backend_manager → _iop → _backend_manager`` import chain that would
    otherwise occur at module load time.
    """
    _set_active_backend(backend)
    from decent_bench.utils.interoperability_2._iop._functions import (  # noqa: PLC0415
        _set_active_backend as _iop_func_set,
    )
    from decent_bench.utils.interoperability_2._iop._rng import (  # noqa: PLC0415
        _set_active_backend as _iop_rng_set,
    )
    _iop_func_set(backend)
    _iop_rng_set(backend)


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
    _notify_backend_subscribers(_NoBackendSet())  # type: ignore[arg-type]


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
    """
    Import the backend's package so its registration side-effect runs.

    Raises:
        ImportError: If the backend module cannot be imported.

    """
    current_module = __name__.rsplit(".", 1)[0]
    module_name = current_module + f"._{backend.value}"
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import the backend module for '{backend.value}'. Ensure the "
            "corresponding backend package is installed and importable."
        ) from exc
