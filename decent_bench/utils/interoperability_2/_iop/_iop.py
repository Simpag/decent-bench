import random
from collections.abc import Sequence
from typing import Any

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability_2._abstracts import _Backend
from decent_bench.utils.interoperability_2._backend_manager import _get_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices, SupportedFrameworks


class _Interoperability(_Backend):  # noqa: PLR0904
    """
    Interoperability class that delegates operations to the selected backend.

    This class implements the _Backend interface and forwards method calls to the active backend instance.
    It serves as the main entry point for interoperability features, allowing users to interact with different
    backends through a unified interface.

    The active backend is determined by the `set_backend` function, which must be called before using any methods
    of this class. Once a backend is set, all method calls will be forwarded to that backend's implementation.

    Example usage:
        from decent_bench.utils.interoperability_2 import Interoperability, set_backend
        from decent_bench.utils.types import SupportedFrameworks

        # Set the active backend
        set_backend(SupportedFrameworks.TORCH)

        # Create an interoperability instance
        iop = Interoperability()

        # Use interoperability methods (these will be forwarded to the active backend)
        rng_state = iop.get_rng_state()
        iop.set_rng_state(rng_state)
        random_array = iop.normal(mean=0.0, std=1.0, shape=(3, 3))

    """

    def __init__(self) -> None:
        self._backend = _get_backend()

        # Need to initialize the numpy as numpy is used for certain RNG operations
        # regardless of whether numpy is selected as the active backend
        self._numpy_backend = _get_backend(SupportedFrameworks.NUMPY)

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return self._backend.zeros(shape)

    def zeros_like(self, array: Array) -> Array:
        return self._backend.zeros_like(array)

    def ones(self, shape: tuple[int, ...]) -> Array:
        return self._backend.ones(shape)

    def ones_like(self, array: Array) -> Array:
        return self._backend.ones_like(array)

    def eye(self, n: int) -> Array:
        return self._backend.eye(n)

    def eye_like(self, array: Array) -> Array:
        return self._backend.eye_like(array)

    def device_to_framework_device(self, device: SupportedDevices, framework: SupportedFrameworks) -> Any:  # noqa: ANN401
        return self._backend.device_to_framework_device(device, framework)

    def framework_device_of_array(self, array: Array) -> Any:  # noqa: ANN401
        return self._backend.framework_device_of_array(array)

    def copy(self, array: Array) -> Array:
        return self._backend.copy(array)

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        return self._backend.stack(arrays, dim)

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self._backend.reshape(array, shape)

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        return self._backend.transpose(array, dim)

    def shape(self, array: Array) -> tuple[int, ...]:
        return self._backend.shape(array)

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        return self._backend.squeeze(array, dim)

    def unsqueeze(self, array: Array, dim: int) -> Array:
        return self._backend.unsqueeze(array, dim)

    def diag(self, array: Array) -> Array:
        return self._backend.diag(array)

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        return self._backend.astype(array, dtype)

    def dot(self, array1: Array, array2: Array) -> Array:
        return self._backend.dot(array1, array2)

    def matmul(self, array1: Array, array2: Array) -> Array:
        return self._backend.matmul(array1, array2)

    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self._backend.norm(array, p, dim, keepdims)

    def sum(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self._backend.sum(array, dim, keepdims)

    def mean(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self._backend.mean(array, dim, keepdims)

    def min(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self._backend.min(array, dim, keepdims)

    def max(
        self,
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self._backend.max(array, dim, keepdims)

    def add(self, array1: Array, array2: Array) -> Array:
        return self._backend.add(array1, array2)

    def iadd[T: Array](self, array1: T, array2: Array) -> T:
        return self._backend.iadd(array1, array2)

    def sub(self, array1: Array, array2: Array) -> Array:
        return self._backend.sub(array1, array2)

    def isub[T: Array](self, array1: T, array2: Array) -> T:
        return self._backend.isub(array1, array2)

    def mul(self, array1: Array, array2: Array) -> Array:
        return self._backend.mul(array1, array2)

    def imul[T: Array](self, array1: T, array2: Array) -> T:
        return self._backend.imul(array1, array2)

    def div(self, array1: Array, array2: Array) -> Array:
        return self._backend.div(array1, array2)

    def idiv[T: Array](self, array1: T, array2: Array) -> T:
        return self._backend.idiv(array1, array2)

    def pow(self, array: Array, p: float) -> Array:
        return self._backend.pow(array, p)

    def ipow[T: Array](self, array: T, p: float) -> T:
        return self._backend.ipow(array, p)

    def negative(self, array: Array) -> Array:
        return self._backend.negative(array)

    def absolute(self, array: Array) -> Array:
        return self._backend.absolute(array)

    def sqrt(self, array: Array) -> Array:
        return self._backend.sqrt(array)

    def sign(self, array: Array) -> Array:
        return self._backend.sign(array)

    def maximum(self, array1: Array, array2: Array) -> Array:
        return self._backend.maximum(array1, array2)

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return self._backend.argmax(array, dim, keepdims)

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return self._backend.argmin(array, dim, keepdims)

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        return self._backend.set_item(array, key, value)

    def get_item(self, array: Array, key: ArrayKey) -> Array:
        return self._backend.get_item(array, key)

    def set_seed(self, seed: int) -> None:
        self._set_seed(seed, set_global_seed=True)

    def _set_seed(self, seed: int, set_global_seed: bool = True) -> None:
        """
        Set random seeds.

        Args:
            seed: Base seed to use.
            set_global_seed: Whether to update the globally tracked seed returned by
                :func:`get_seed`. Set this to ``False`` for trial-local reseeding where
                preserving the external base seed is required.

        """
        random.seed(seed)
        self._numpy_backend.set_seed(seed)
        self._backend.set_seed(seed)

        if set_global_seed:
            self.global_seed = seed

    def get_seed(self) -> int:
        """
        Get the current global random seed.

        Returns:
            int: The current global random seed.

        """
        return self.global_seed

    def get_rng_state(self) -> dict[str, Any]:
        if self._backend is self._numpy_backend:
            return self._backend.get_rng_state()
        # If the active backend is not numpy, we need to get the RNG state from both the active backend and numpy
        # to ensure that we can fully restore the RNG state later. This is because some operations may use numpy's
        # RNG even when a different backend is active.
        state = self._backend.get_rng_state()
        state["__numpy_rng_state__"] = self._numpy_backend.get_rng_state()
        return state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if self._backend is not self._numpy_backend and "__numpy_rng_state__" in state:
            # If the active backend is not numpy, we need to set the RNG state for both the active backend and numpy
            # to ensure that we can fully restore the RNG state. This is because some operations may use numpy's
            # RNG even when a different backend is active.
            self._numpy_backend.set_rng_state(state.pop("__numpy_rng_state__"))
        self._backend.set_rng_state(state)

    def normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        shape: tuple[int, ...] = (),
    ) -> Array:
        return self._backend.normal(mean, std, shape)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] = (),
    ) -> Array:
        return self._backend.uniform(low, high, shape)

    def normal_like(self, array: Array, mean: float = 0, std: float = 1) -> Array:
        return self._backend.normal_like(array, mean, std)

    def uniform_like(self, array: Array, low: float = 0, high: float = 1) -> Array:
        return self._backend.uniform_like(array, low, high)

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        return self._backend.choice(array, size, replace)
