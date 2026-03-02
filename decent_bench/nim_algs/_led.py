from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import decent_bench.utils.algorithm_helpers as alg_helpers
import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import P2PNetwork

if TYPE_CHECKING:
    from decent_bench.utils.array import Array


@dataclass(eq=False)
class LED(Algorithm):
    """
    Local Exact-Diffusion (LED) algorithm :footcite:p:`LED_Alghunaim_2024`.

    Args:
        iterations: Total number of communication rounds (r)
        local_steps: Number of local updates (tau)
        step_size: Step size alpha for gradient steps, can be a constant or a function of iteration
        aux_step_size: Step size beta for dual variable, can be a constant or a function of iteration
        x0: Initial parameters (optional)
        name: Algorithm name (default "LED")

    """

    iterations: int = 100  # Total number of communication rounds (r)
    local_steps: int = 5  # Number of local updates (tau)
    step_size: float | Callable[[int], float] = 0.01  # Step size alpha for gradient steps
    aux_step_size: float | Callable[[int], float] = 0.01  # Step size beta for dual variable
    x0: "Array | None" = None  # Initial parameters (optional)
    name: str = "LED"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if callable(self.step_size):
            test_step_size = [self.step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_step_size):
                raise ValueError("step_size function must return positive values for all iterations")
        if isinstance(self.aux_step_size, float) and self.aux_step_size <= 0:
            raise ValueError("aux_step_size must be positive")
        if callable(self.aux_step_size):
            test_aux_step_size = [self.aux_step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_aux_step_size):
                raise ValueError("aux_step_size function must return positive values for all iterations")

    def initialize(self, network: P2PNetwork) -> None:
        """Initialize agents with x_i^0, y_i^0, and phi_i,0^r."""
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        self.W = network.weights

        for agent in network.agents():
            # Initialize y_i^0 = 0 (simplified initialization)
            y_0 = iop.zeros_like(self.x0)

            # Initialize auxiliary variables
            aux_vars = {
                "y": y_0,  # Dual variable y_i^r
                "phi": self.x0,  # phi_i,tau^r (to be broadcasted)
            }

            agent.initialize(
                x=self.x0,
                aux_vars=aux_vars,
                received_msgs=dict.fromkeys(network.neighbors(agent), self.x0),
            )

    def step(self, network: P2PNetwork, iteration: int) -> None:
        step_size = self.step_size(iteration) if callable(self.step_size) else self.step_size
        aux_step_size = self.aux_step_size(iteration) if callable(self.aux_step_size) else self.aux_step_size

        # Step 1: Local primal updates (tau steps)
        for i in network.active_agents(iteration):
            self._local_primal_updates(i, step_size, aux_step_size)

        # Step 2: Diffusion (communication and mixing)
        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["phi"])

        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            self._diffusion(i)

        # Step 3: Local dual update
        for i in network.active_agents(iteration):
            self._local_dual_update(i)

    def _local_primal_updates(self, agent: Agent, step_size: float, aux_step_size: float) -> None:
        """
        Step 1: Local primal updates (tau steps).

        Algorithm 1, line 1:
        """
        # Set phi_i,0^r = x_i^r (line 1)
        agent.aux_vars["phi"] = iop.copy(agent.x)

        # Perform tau local updates (Equation 2a)
        for _ in range(self.local_steps):
            gradient = agent.cost.gradient(agent.aux_vars["phi"])
            agent.aux_vars["phi"] -= step_size * gradient + aux_step_size * agent.aux_vars["y"]

    def _diffusion(self, agent: Agent) -> None:
        """
        Step 2: Diffusion.

        Algorithm 1, line 2:
        """
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * phi_j for j, phi_j in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * agent.aux_vars["phi"]
        agent.x = weighted_sum

    def _local_dual_update(self, agent: Agent) -> None:
        """
        Step 3: Local dual update.

        Algorithm 1, line 3:

        Update the dual variable for exact tracking (Equation 2c).
        """
        agent.aux_vars["y"] += agent.aux_vars["phi"] - agent.x
