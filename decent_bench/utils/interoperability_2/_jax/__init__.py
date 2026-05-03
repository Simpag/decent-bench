"""JAX backend package; importing it triggers backend registration."""

from ._jax_backend import JaxBackend

__all__ = ["JaxBackend"]
