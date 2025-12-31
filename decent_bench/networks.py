from __future__ import annotations

from collections.abc import Collection, Sequence
from functools import cached_property
from typing import Any, cast

import networkx as nx
import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.abstracts.cost import Cost
from decent_bench.abstracts.network import AgentGraph, Network
from decent_bench.abstracts.scheme import CompressionScheme, DropScheme, NoiseScheme
from decent_bench.agents import Agent
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.utils.array import Array


class P2PNetwork[CF: Cost](Network[CF]):
    """Peer-to-peer network architecture where agents communicate directly with each other."""

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ) -> None:
        super().__init__(
            graph=graph,
            message_noise=message_noise,
            message_compression=message_compression,
            message_drop=message_drop,
        )
        self.W: Array | None = None

    @property
    def weights(self) -> Array:
        """
        Symmetric, doubly stochastic matrix for consensus weights. Initialized using the Metropolis-Hastings method.

        Use ``weights[i, j]`` or ``weights[i.id, j.id]`` to get the weight between agent i and j.
        """
        agents = self.agents()

        if self.W is not None:
            return self.W

        n = len(agents)
        W = np.zeros((n, n))  # noqa: N806
        degrees = self.degrees
        for i in agents:
            neighbors = self.neighbors(i)
            d_i = degrees[i]
            for j in neighbors:
                d_j = degrees[j]
                W[i, j] = 1 / (1 + max(d_i, d_j))
        for i in agents:
            W[i, i] = 1 - sum(W[i])

        self.W = iop.to_array(W, agents[0].cost.framework, agents[0].cost.device)
        return self.W

    @weights.setter
    def weights(self, weights: Array) -> None:
        """
        Set custom consensus weights matrix.

        A simple way to create custom weights is to start using numpy and then
        use :func:`~decent_bench.utils.interoperability.to_array` to convert to an
        :class:`~decent_bench.utils.array.Array` object with the desired framework and device.
        For an example see :func:`~decent_bench.utils.interoperability.zeros`.

        Note:
            If not set, the weights matrix is initialized using the Metropolis-Hastings method.

        Raises:
            ValueError: if the weights matrix does not have shape (n_agents, n_agents)
            ValueError: if the weights matrix does not have the same framework and device as the agents' cost functions

        """
        if iop.shape(weights) != (len(self.agents()), len(self.agents())):
            raise ValueError("Weights matrix must have shape (n_agents, n_agents)")

        if iop.framework_device_of_array(weights) != (self.agents()[0].cost.framework, self.agents()[0].cost.device):
            raise ValueError("Weights matrix must have the same framework and device as the agents' cost functions")

        self.W = weights

    @cached_property
    def adjacency(self) -> Array:
        """
        Adjacency matrix of the network.

        Use ``adjacency[i, j]`` or ``adjacency[i.id, j.id]`` to get the adjacency between agent i and j.
        """
        agents = self.agents()
        adjacency_matrix = nx.to_numpy_array(
            self.graph,
            nodelist=cast("Collection[Any]", agents),
            dtype=float,  # pyright: ignore[reportArgumentType]
        )  # type: ignore[call-overload]
        return iop.to_array(
            adjacency_matrix,
            agents[0].cost.framework,
            agents[0].cost.device,
        )

    def neighbors(self, agent: Agent[CF]) -> list[Agent[CF]]:
        """Alias for :meth:`~decent_bench.networks.Network.connected_agents`."""
        return super().connected_agents(agent)

    def broadcast(self, sender: Agent[CF], msg: Array) -> None:
        """Send to all neighbors (alias for :meth:`~decent_bench.networks.Network.send` with ``receiver=None``)."""
        self.send(sender=sender, receiver=None, msg=msg)

    def receive_all(self, receiver: Agent[CF]) -> None:
        """Receive from all neighbors (alias for Network.receive with sender=None)."""
        self.receive(receiver=receiver, sender=None)


class FedNetwork[CF: Cost](Network[CF]):
    """Federated learning network with one server node connected to all client nodes (star topology)."""

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ) -> None:
        super().__init__(
            graph=graph,
            message_noise=message_noise,
            message_compression=message_compression,
            message_drop=message_drop,
        )
        self._server = self._identify_server()

    def _identify_server(self) -> Agent[CF]:
        degrees = dict(self.graph.degree())
        if not degrees:
            raise ValueError("FedNetwork requires at least one agent")
        server, max_degree = max(degrees.items(), key=lambda item: item[1])  # noqa: FURB118
        n = len(degrees)
        if max_degree != n - 1 or any(deg != 1 for node, deg in degrees.items() if node != server):
            raise ValueError("FedNetwork expects a star topology with one server connected to all clients")
        return server

    @property
    def server(self) -> Agent[CF]:
        """Agent acting as the central server."""
        return self._server

    @property
    def coordinator(self) -> Agent[CF]:
        """Alias for :attr:`server`."""
        return self.server

    def agents(self) -> list[Agent[CF]]:
        """Get all client agents (excludes the server/coordinator)."""
        return [agent for agent in self.graph if agent is not self.server]

    def active_agents(self, iteration: int) -> list[Agent[CF]]:
        """Get all active client agents (excludes the server/coordinator)."""
        # Delegates to Network.active_agents(), which iterates over self.agents() (clients only for FedNetwork).
        return super().active_agents(iteration)

    @property
    def clients(self) -> list[Agent[CF]]:
        """Alias for :meth:`agents`."""
        return self.agents()

    def active_clients(self, iteration: int) -> list[Agent[CF]]:
        """Alias for :meth:`active_agents`."""
        return self.active_agents(iteration)

    def send(
        self,
        sender: Agent[CF],
        receiver: Agent[CF] | Sequence[Agent[CF]] | None = None,
        msg: Array | None = None,
    ) -> None:
        """
        Send message(s) in a federated learning network.

        Only server <-> client communication is allowed. Client-to-client and server-to-server communication will
        raise an error.

        Raises:
            ValueError: if server-to-server or client-to-client communication is attempted, or if a non-server tries to
                send to multiple receivers. Also see :meth:`Network.send` for generic validation.

        """
        if isinstance(receiver, Agent):
            if sender is self.server and receiver is self.server:
                raise ValueError("Server-to-server communication is not supported")
            if sender is not self.server and receiver is not self.server:
                raise ValueError("Client-to-client communication is not supported")
            super().send(sender=sender, receiver=receiver, msg=msg)
            return

        if receiver is None:
            super().send(sender=sender, receiver=receiver, msg=msg)
            return

        if sender is not self.server:
            raise ValueError("Only the server can send to multiple receivers")
        if any(r is self.server for r in receiver):
            raise ValueError("All receivers must be clients")
        super().send(sender=sender, receiver=receiver, msg=msg)

    def receive(self, receiver: Agent[CF], sender: Agent[CF] | Sequence[Agent[CF]] | None = None) -> None:
        """
        Receive message(s) in a federated learning network.

        Only server <-> client communication is allowed. Client-to-client and server-to-server communication will
        raise an error.

        Raises:
            ValueError: if sender/receiver roles are invalid. Also see :meth:`Network.receive` for generic validation.

        """
        if isinstance(sender, Agent):
            if receiver is self.server and sender is self.server:
                raise ValueError("Server-to-server communication is not supported")
            if receiver is not self.server and sender is not self.server:
                raise ValueError("Client-to-client communication is not supported")
            super().receive(receiver=receiver, sender=sender)
            return

        if sender is None:
            super().receive(receiver=receiver, sender=sender)
            return

        if receiver is not self.server:
            raise ValueError("Only the server can receive from multiple senders")
        if any(s is self.server for s in sender):
            raise ValueError("All senders must be clients")
        super().receive(receiver=receiver, sender=sender)

    def broadcast(self, msg: Array) -> None:
        """Send the same message from the server to every client (synchronous FL push)."""
        self.send(sender=self.server, receiver=None, msg=msg)

    def receive_all(self) -> None:
        """Receive messages at the server from every client (synchronous FL pull)."""
        self.receive(receiver=self.server, sender=None)


def create_distributed_network[CF: Cost](problem: BenchmarkProblem[CF]) -> P2PNetwork[CF]:
    """
    Create a distributed network - a network with peer-to-peer communication only, no coordinator.

    Raises:
        ValueError: if there are less agent activation schemes or cost functions than agents

    """
    n_agents = len(problem.network_structure)
    if len(problem.agent_activations) < n_agents:
        raise ValueError("Insufficient number of agent activation schemes, please provide one per agent")
    if len(problem.costs) < n_agents:
        raise ValueError("Insufficient number of cost functions, please provide one per agent")
    if problem.network_structure.is_directed():
        raise NotImplementedError("Support for directed graphs has not been implemented yet")
    if problem.network_structure.is_multigraph():
        raise NotImplementedError("Support for multi-graphs has not been implemented yet")
    if not nx.is_connected(problem.network_structure):
        raise NotImplementedError("Support for disconnected graphs has not been implemented yet")
    agents = [
        Agent(i, problem.costs[i], problem.agent_activations[i], problem.agent_state_snapshot_period)
        for i in range(n_agents)
    ]
    agent_node_map = {node: agents[i] for i, node in enumerate(problem.network_structure.nodes())}
    graph = nx.relabel_nodes(problem.network_structure, agent_node_map)
    return P2PNetwork(
        graph=graph,
        message_noise=problem.message_noise,
        message_compression=problem.message_compression,
        message_drop=problem.message_drop,
    )


def create_federated_network[CF: Cost](problem: BenchmarkProblem[CF]) -> FedNetwork[CF]:
    """
    Create a federated learning network with a single server and multiple clients (star topology).

    Raises:
        ValueError: if there are fewer activation schemes or cost functions than agents
        ValueError: if the provided graph is not a star (one server connected to all clients)

    """
    n_agents = len(problem.network_structure)
    if len(problem.agent_activations) < n_agents:
        raise ValueError("Insufficient number of agent activation schemes, please provide one per agent")
    if len(problem.costs) < n_agents:
        raise ValueError("Insufficient number of cost functions, please provide one per agent")
    if problem.network_structure.is_directed():
        raise NotImplementedError("Support for directed graphs has not been implemented yet")
    if problem.network_structure.is_multigraph():
        raise NotImplementedError("Support for multi-graphs has not been implemented yet")
    if not nx.is_connected(problem.network_structure):
        raise NotImplementedError("Support for disconnected graphs has not been implemented yet")
    degrees = dict(problem.network_structure.degree())
    if n_agents:
        server, max_degree = max(degrees.items(), key=lambda item: item[1])  # noqa: FURB118
        if max_degree != n_agents - 1 or any(deg != 1 for node, deg in degrees.items() if node != server):
            raise ValueError("Federated network requires a star topology (one server connected to all clients)")
    agents = [
        Agent(i, problem.costs[i], problem.agent_activations[i], problem.agent_state_snapshot_period)
        for i in range(n_agents)
    ]
    agent_node_map = {node: agents[i] for i, node in enumerate(problem.network_structure.nodes())}
    graph = nx.relabel_nodes(problem.network_structure, agent_node_map)
    return FedNetwork(
        graph=graph,
        message_noise=problem.message_noise,
        message_compression=problem.message_compression,
        message_drop=problem.message_drop,
    )
