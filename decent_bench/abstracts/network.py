from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx

from decent_bench.abstracts.cost import Cost
from decent_bench.abstracts.scheme import CompressionScheme, DropScheme, NoiseScheme
from decent_bench.agents import Agent
from decent_bench.utils.array import Array

if TYPE_CHECKING:
    AgentGraph = nx.Graph[Agent[Any]]
else:
    AgentGraph = nx.Graph


class Network[CF: Cost](ABC):  # noqa: B024
    """Base network object defining communication logic shared by all network types."""

    def __init__(
        self,
        graph: AgentGraph,
        message_noise: NoiseScheme,
        message_compression: CompressionScheme,
        message_drop: DropScheme,
    ) -> None:
        self._graph = graph
        self._message_noise = message_noise
        self._message_compression = message_compression
        self._message_drop = message_drop

    @property
    def graph(self) -> AgentGraph:
        """Underlying NetworkX graph; mutating it will change the network."""
        return self._graph

    @property
    def G(self) -> AgentGraph:  # noqa: N802
        """Alias for the underlying graph."""
        return self.graph

    def agents(self) -> list[Agent[CF]]:
        """Get all agents in the network."""
        return list(self.graph)

    @property
    def degrees(self) -> dict[Agent[CF], int]:
        """Degree of each agent in the network."""
        return dict(self.graph.degree())

    @property
    def edges(self) -> list[tuple[Agent[CF], Agent[CF]]]:
        """Edges of the network as (agent, agent) tuples."""
        return list(self.graph.edges())

    def active_agents(self, iteration: int) -> list[Agent[CF]]:
        """
        Get all active agents.

        Whether an :class:`~decent_bench.agents.Agent` is active or not at a given time is defined by its
        :class:`~decent_bench.schemes.AgentActivationScheme`.
        """
        return [a for a in self.agents() if a._activation.is_active(iteration)]  # noqa: SLF001

    def connected_agents(self, agent: Agent[CF]) -> list[Agent[CF]]:
        """Agents directly connected to ``agent`` in the underlying graph."""
        return list(self.graph.neighbors(agent))

    def _send_one(self, sender: Agent[CF], receiver: Agent[CF], msg: Array) -> None:
        """
        Send message to an agent.

        The message may be compressed, distorted by noise, and/or dropped depending on the network's
        :class:`~decent_bench.schemes.CompressionScheme`,
        :class:`~decent_bench.schemes.NoiseScheme`,
        and :class:`~decent_bench.schemes.DropScheme`.

        The message will stay in-flight until it is received or replaced by a newer message from the same sender to the
        same receiver. After being received or replaced, the message is destroyed.
        """
        sender._n_sent_messages += 1  # noqa: SLF001
        if self._message_drop.should_drop():
            sender._n_sent_messages_dropped += 1  # noqa: SLF001
            return
        msg = self._message_compression.compress(msg)
        msg = self._message_noise.make_noise(msg)
        self.graph.edges[sender, receiver][str(receiver.id)] = msg

    def send(
        self,
        sender: Agent[CF],
        receiver: Agent[CF] | Sequence[Agent[CF]] | None = None,
        msg: Array | None = None,
    ) -> None:
        """
        Send message to one or more agents.

        Args:
            sender: sender agent
            receiver: receiver agent, sequence of receiver agents, or ``None`` to broadcast to connected agents.
            msg: message to send

        Raises:
            ValueError: if ``msg`` is not provided, if agents are not part of the network, or if sender/receiver are not
                connected.

        """
        if msg is None:
            raise ValueError("msg must be provided")

        if sender not in self.graph:
            raise ValueError("Sender must be an agent in the network")

        if receiver is None:
            receiver = self.connected_agents(sender)
        elif isinstance(receiver, Agent):
            if receiver not in self.connected_agents(sender):
                raise ValueError("Sender and receiver must be connected in the network")
            self._send_one(sender=sender, receiver=receiver, msg=msg)
            return
        neighbors = set(self.connected_agents(sender))
        invalid_receivers = [r for r in receiver if r not in neighbors]
        if invalid_receivers:
            ids = [r.id for r in invalid_receivers]
            raise ValueError(f"Sender and receiver must be connected in the network; not connected receivers: {ids}")

        for r in receiver:
            self._send_one(sender=sender, receiver=r, msg=msg)

    def _receive_one(self, receiver: Agent[CF], sender: Agent[CF]) -> None:
        """
        Receive message from an agent.

        Received messages are stored in
        :attr:`Agent.messages <decent_bench.agents.Agent.messages>`.
        """
        msg = self.graph.edges[sender, receiver].get(str(receiver.id))
        if msg is not None:
            receiver._n_received_messages += 1  # noqa: SLF001
            receiver._received_messages[sender] = msg  # noqa: SLF001
            self.graph.edges[sender, receiver][str(receiver.id)] = None

    def receive(self, receiver: Agent[CF], sender: Agent[CF] | Sequence[Agent[CF]] | None = None) -> None:
        """
        Receive message(s) at an agent.

        Args:
            receiver: receiver agent
            sender: sender agent, sequence of sender agents, or ``None`` to receive from all connected agents.

        Raises:
            ValueError: if sender/receiver are not part of the network or not connected.

        """
        if receiver not in self.graph:
            raise ValueError("Receiver must be an agent in the network")

        if sender is None:
            sender = self.connected_agents(receiver)
        elif isinstance(sender, Agent):
            if sender not in self.connected_agents(receiver):
                raise ValueError("Sender and receiver must be connected in the network")
            self._receive_one(receiver=receiver, sender=sender)
            return
        neighbors = set(self.connected_agents(receiver))
        invalid_senders = [s for s in sender if s not in neighbors]
        if invalid_senders:
            ids = [s.id for s in invalid_senders]
            raise ValueError(f"Sender and receiver must be connected in the network; not connected senders: {ids}")

        for s in sender:
            self._receive_one(receiver=receiver, sender=s)
