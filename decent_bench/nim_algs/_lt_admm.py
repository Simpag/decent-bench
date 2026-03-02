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
class LT_ADMM(Algorithm):  # noqa: N801
    """
    Local Training ADMM (LT-ADMM) :footcite:p:`LT_ADMM_VR_Ren_2026`.

    Args:
        iterations: Total number of communication rounds (K)
        local_steps: Number of local training steps (tau)
        step_size: Local step size (gamma), can be a constant or a function of iteration
        penalty: Penalty parameter (rho)
        alpha: Relaxation parameter (alpha)
        x0: Initial parameters (optional)
        name: Algorithm name (default "LT-ADMM")

    """

    iterations: int = 100  # Total number of communication rounds (K)
    local_steps: int = 5  # Number of local training steps (tau)
    step_size: float | Callable[[int], float] = 0.01  # Local step size (gamma)
    penalty: float = 1.0  # Penalty parameter (rho)
    alpha: float = 0.5  # Relaxation parameter (alpha)
    x0: "Array | None" = None  # Initial parameters (optional)
    name: str = "LT-ADMM"

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
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def initialize(self, network: P2PNetwork) -> None:
        self.x0 = alg_helpers.zero_initialization(self.x0, network)

        # Initialize agents with auxiliary variables
        for i in network.agents():
            neighbors = network.neighbors(i)
            z_i = iop.zeros(
                (len(neighbors), *iop.shape(self.x0)),
                framework=i.cost.framework,
                device=i.cost.device,
            )
            neighbor_to_idx = {}

            for idx, j in enumerate(neighbors):
                z_i[idx] = iop.copy(self.x0)
                neighbor_to_idx[j] = idx

            aux_vars = {
                "phi": self.x0,  # phi_i,k - model parameters
                "z_i": z_i,  # z_ij,k+1 - auxiliary consensus variable
                "neighbor_to_idx": neighbor_to_idx,
            }
            i.initialize(x=self.x0, aux_vars=aux_vars, received_msgs=dict.fromkeys(neighbors, self.x0))

    def step(self, network: P2PNetwork, iteration: int) -> None:
        step_size = self.step_size(iteration) if callable(self.step_size) else self.step_size

        # Step 1: Local training phase
        for i in network.active_agents(iteration):
            self._local_training(i, network, step_size)

        # Step 2: Communication phase
        for i in network.active_agents(iteration):
            self._communication(i, network)

        # Step 3: Auxiliary update phase
        for i in network.active_agents(iteration):
            self._auxiliary_update(i, network)

    def _local_training(self, agent: Agent, network: P2PNetwork, step_size: float) -> None:
        """
        Perform local training steps (Algorithm 1, lines 2-9).

        Updates phi_i,k and gradient estimators r_i,h,k.
        """
        neighbors = network.neighbors(agent)

        agent.aux_vars["phi"] = iop.copy(agent.x)
        z_sum = iop.sum(agent.aux_vars["z_i"], dim=0)

        for _ in range(self.local_steps):
            current_gradient = agent.cost.gradient(agent.aux_vars["phi"])
            step = current_gradient + self.penalty * len(neighbors) * agent.aux_vars["phi"] - z_sum
            # Update phi_i,k according to gradient step (line 7)
            agent.aux_vars["phi"] -= step_size * step

        # Update agent's main parameter (line 10)
        agent.x = agent.aux_vars["phi"]

    def _communication(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Communication phase (Algorithm 1, line 11).

        Transmit z_ij,k - 2 * rho * x_i,k+1 to each neighbor.
        """
        # Transmit z_i,k - 2 * rho * x_i,k+1 to each neighbor j in N_i
        for j in network.neighbors(agent):
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            message = agent.aux_vars["z_i"][j_idx] - 2 * self.penalty * agent.x
            network.send(agent, j, message)

    def _auxiliary_update(self, agent: Agent, network: P2PNetwork) -> None:
        """
        Auxiliary update phase (Algorithm 1, line 12).

        Update z_ij,k+1 according to equation (3b).
        """
        network.receive_all(agent)

        for j, msg in agent.messages.items():
            j_idx = agent.aux_vars["neighbor_to_idx"][j]
            z_update = (1 - self.alpha) * agent.aux_vars["z_i"][j_idx] - self.alpha * msg
            agent.aux_vars["z_i"][j_idx] = z_update
