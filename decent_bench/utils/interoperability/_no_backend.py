"""
No-op backend for interoperability_2.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("numpy")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from decent_bench.utils.interoperability._abstracts._backend import _Backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from decent_bench.utils.array import Array
    from decent_bench.utils.types import ArrayKey, SupportedDevices


class _NoBackend(_Backend):  # noqa: PLR0904
    """No-op implementation of :class:`_Backend`."""

    # Array creation

    def zeros(self, shape: tuple[int, ...]) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def zeros_like(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def ones(self, shape: tuple[int, ...]) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def ones_like(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def eye(self, n: int) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def eye_like(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def device_to_native(self, device: SupportedDevices) -> Any:  # noqa: ANN401, ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def device_of(self, array: Array) -> SupportedDevices:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    # Array manipulation

    def copy(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def to_numpy(self, array: Array) -> NDArray[Any]:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def from_numpy(self, array: NDArray[Any]) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def shape(self, array: Array) -> tuple[int, ...]:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def unsqueeze(self, array: Array, dim: int) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def diag(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    # Linalg

    def dot(self, array1: Array, array2: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def matmul(self, array1: Array, array2: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def norm(
        self,
        array: Array,  # noqa: ARG002
        p: float = 2,  # noqa: ARG002
        dim: int | tuple[int, ...] | None = None,  # noqa: ARG002
        keepdims: bool = False,  # noqa: ARG002
    ) -> Array:
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    # Math reductions

    def sum(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def mean(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def min(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def max(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, array1: Array | float, array2: Array | float) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def iadd[T: Array](self, array1: T, array2: Array | float) -> T:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def sub(self, array1: Array | float, array2: Array | float) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def isub[T: Array](self, array1: T, array2: Array | float) -> T:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def mul(self, array1: Array | float, array2: Array | float) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def imul[T: Array](self, array1: T, array2: Array | float) -> T:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def div(self, array1: Array | float, array2: Array | float) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def idiv[T: Array](self, array1: T, array2: Array | float) -> T:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def pow(self, array: Array, p: float) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def negative(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def absolute(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def sqrt(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    # Operators

    def sign(self, array: Array) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def maximum(self, array1: Array | float, array2: Array | float) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def get_item(self, array: Array, key: ArrayKey) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    # RNG

    def set_seed(self, seed: int) -> None:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def get_rng_state(self) -> dict[str, Any]:
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def set_rng_state(self, state: dict[str, Any]) -> None:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:  # noqa: ARG002
        raise RuntimeError("No active backend. Call set_backend(...) first.")
