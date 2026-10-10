"""Small, inspectable RL agents and observation representations.

The representations in this module deliberately operate only on the public
observation dictionaries emitted by :mod:`src.environment`. They cannot see
the source location, grid coordinates, or other privileged environment state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Observation = Mapping[str, int]
State = Hashable


def observation_token(observation: Observation) -> tuple[tuple[str, int], ...]:
    """Encode an observation as a stable, hashable public-information token."""
    if not observation:
        raise ValueError("An agent observation must contain at least one sensor value.")
    return tuple(sorted((str(key), int(value)) for key, value in observation.items()))


def observation_vector(
    observation: Observation,
    history: Sequence[Observation] | None = None,
    history_window: int = 1,
) -> np.ndarray:
    """Encode recent public observations into a fixed-length numeric vector.

    The vector is intentionally compact: odor is carried as the raw sensor value,
    wind is normalized to the 0..3 direction code, and previous_action is
    normalized by the action count. This keeps the DQN input interpretable while
    remaining compatible with a small feed-forward Q-network.
    """
    if int(history_window) < 1:
        raise ValueError("history_window must be at least 1.")
    window = int(history_window)
    recent = list(history) if history is not None else [observation]
    payload = np.zeros((window, 3), dtype=np.float32)
    for index, sample in enumerate(recent[:window]):
        payload[index, 0] = float(sample.get("odor", 0))
        payload[index, 1] = float(sample.get("wind", 0)) / 3.0
        payload[index, 2] = (
            float(sample.get("previous_action", 0)) / 4.0
            if "previous_action" in sample
            else 0.0
        )
    if len(recent) < window:
        for index in range(len(recent), window):
            payload[index] = 0.0
    return payload.reshape(-1)


class StateRepresentation(ABC):
    """Interface separating sensory context from the Q-learning algorithm."""

    name: str

    @abstractmethod
    def reset(self, observation: Observation) -> None:
        """Clear episode-local context and record the reset observation."""

    @abstractmethod
    def update(self, observation: Observation) -> None:
        """Incorporate one newly received public observation."""

    @abstractmethod
    def state(self) -> State:
        """Return a hashable state based only on incorporated observations."""

    @abstractmethod
    def latest_token(self) -> tuple[tuple[str, int], ...]:
        """Return the newest token so callers can guard against stale context."""


class CurrentObservationRepresentation(StateRepresentation):
    """Agent A: state is exactly the latest permitted observation."""

    name = "current_observation"

    def __init__(self) -> None:
        self._latest: tuple[tuple[str, int], ...] | None = None

    def reset(self, observation: Observation) -> None:
        self._latest = observation_token(observation)

    def update(self, observation: Observation) -> None:
        self._latest = observation_token(observation)

    def state(self) -> State:
        if self._latest is None:
            raise RuntimeError("Call reset_episode() before selecting an action.")
        return self._latest

    def latest_token(self) -> tuple[tuple[str, int], ...]:
        return self.state()


class HistoryRepresentation(StateRepresentation):
    """Agent B: fixed-length sequence of public observation tokens."""

    name = "sensory_history"
    _PAD_TOKEN: tuple[tuple[str, int], ...] = (("__episode_padding__", -1),)

    def __init__(self, window_size: int = 4) -> None:
        if int(window_size) < 1:
            raise ValueError("history window_size must be at least 1.")
        self.window_size = int(window_size)
        self._history: deque[tuple[tuple[str, int], ...]] = deque(maxlen=self.window_size)

    def reset(self, observation: Observation) -> None:
        self._history.clear()
        self._history.append(observation_token(observation))

    def update(self, observation: Observation) -> None:
        self._history.append(observation_token(observation))

    def state(self) -> State:
        if not self._history:
            raise RuntimeError("Call reset_episode() before selecting an action.")
        padding = [self._PAD_TOKEN] * (self.window_size - len(self._history))
        return tuple(padding + list(self._history))

    def latest_token(self) -> tuple[tuple[str, int], ...]:
        if not self._history:
            raise RuntimeError("Call reset_episode() before selecting an action.")
        return self._history[-1]


class GenericContextRepresentation(StateRepresentation):
    """Optional Agent C placeholder: compressed context from recent sensors.

    This is explicitly a generic experimental baseline, *not* ACIL. It records
    a rolling sensor history, then exposes a compact feature tuple derived from
    the recent public observations. This placeholder supports future context
    comparisons but does not claim any ACIL behavior.
    """

    name = "generic_context_placeholder_not_acil"

    def __init__(self, window_size: int = 4) -> None:
        if int(window_size) < 1:
            raise ValueError("context window_size must be at least 1.")
        self.window_size = int(window_size)
        self._history: deque[dict[str, int]] = deque(maxlen=self.window_size)

    def reset(self, observation: Observation) -> None:
        self._history.clear()
        self.update(observation)

    def update(self, observation: Observation) -> None:
        self._history.append({str(key): int(value) for key, value in observation.items()})

    def state(self) -> State:
        if not self._history:
            raise RuntimeError("Call reset_episode() before selecting an action.")
        observations = list(self._history)
        latest = observation_token(observations[-1])
        odor_count = sum(int(item.get("odor", 0) == 1) for item in observations)
        wind_changes = sum(
            int(before.get("wind") != after.get("wind"))
            for before, after in zip(observations, observations[1:])
            if "wind" in before and "wind" in after
        )
        action_changes = sum(
            int(before.get("previous_action") != after.get("previous_action"))
            for before, after in zip(observations, observations[1:])
            if "previous_action" in before and "previous_action" in after
        )
        return (
            "generic_context",
            latest,
            odor_count,
            wind_changes,
            action_changes,
        )

    def latest_token(self) -> tuple[tuple[str, int], ...]:
        if not self._history:
            raise RuntimeError("Call reset_episode() before selecting an action.")
        return observation_token(self._history[-1])


def build_representation(method: str, history_window: int) -> StateRepresentation:
    """Create a representation by the stable experiment method identifier."""
    normalized = method.strip().lower()
    if normalized == "current_observation":
        return CurrentObservationRepresentation()
    if normalized == "sensory_history":
        return HistoryRepresentation(history_window)
    if normalized in {"generic_context", "generic_context_placeholder_not_acil"}:
        return GenericContextRepresentation(history_window)
    raise ValueError(
        "Unknown method {!r}. Use current_observation, sensory_history, or "
        "generic_context.".format(method)
    )


class DQNNetwork(nn.Module):
    """Compact feed-forward Q-network for a bounded discrete action space."""

    def __init__(self, input_dim: int, action_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    """Genuine DQN policy for current-observation and history-based conditions."""

    def __init__(
        self,
        action_count: int = 5,
        history_window: int = 1,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        target_update_interval: int = 200,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if int(action_count) < 1:
            raise ValueError("action_count must be positive.")
        if int(history_window) < 1:
            raise ValueError("history_window must be at least 1.")
        self.action_count = int(action_count)
        self.history_window = int(history_window)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or "cpu")
        self.state_dim = self.history_window * 3
        self.network = DQNNetwork(self.state_dim, self.action_count).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_count).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.replay: deque[tuple[np.ndarray, np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.replay_capacity
        )
        self.training_steps = 0
        self.history: deque[Observation] = deque(maxlen=self.history_window)
        self._latest_state: np.ndarray | None = None

    @property
    def representation_name(self) -> str:
        if self.history_window == 1:
            return "current_observation_dqn"
        return "sensory_history_dqn"

    def _normalize_observation(self, observation: Observation) -> Observation:
        normalized: dict[str, int] = {}
        if "odor" in observation:
            normalized["odor"] = int(observation["odor"])
        if "wind" in observation:
            normalized["wind"] = int(observation["wind"]) % 4
        if "previous_action" in observation:
            normalized["previous_action"] = int(observation["previous_action"])
        return normalized

    def reset_episode(self, observation: Observation) -> None:
        self.history.clear()
        self.history.append(self._normalize_observation(observation))
        self._latest_state = self.current_state_vector()

    def advance_history(self, observation: Observation) -> None:
        self.history.append(self._normalize_observation(observation))
        self._latest_state = self.current_state_vector()

    def current_state_vector(self) -> np.ndarray:
        if not self.history:
            raise RuntimeError("Call reset_episode() before selecting an action.")
        return observation_vector(
            observation=dict(self.history[-1]),
            history=list(self.history),
            history_window=self.history_window,
        )

    def act(self, observation: Observation, explore: bool = True) -> int:
        """Choose an action from the current public history without mutating it.

        The caller is responsible for appending the next observation after the
        environment transition, which keeps the history consistent across
        training, evaluation, and checkpoint restoration.
        """
        state = self.current_state_vector()
        if explore and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, self.action_count))
            return action
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.network(tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.append((state.copy(), next_state.copy(), int(action), float(reward), done))

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        transitions = [self.replay[idx] for idx in batch]

        states = torch.as_tensor(np.stack([t[0] for t in transitions]), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t[1] for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t[2] for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t[3] for t in transitions], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(t[4]) for t in transitions], dtype=torch.float32, device=self.device)

        q_values = self.network(states)
        action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.mse_loss(action_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.training_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return float(loss.item())

    def save(self, path: str | "PathLike[str]" | None) -> None:
        if path is None:
            raise ValueError("A checkpoint path is required.")
        checkpoint = {
            "network_state": self.network.state_dict(),
            "target_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "history_window": self.history_window,
            "action_count": self.action_count,
        }
        torch.save(checkpoint, path)

    def load(self, path: str | "PathLike[str]" | None) -> None:
        if path is None:
            raise ValueError("A checkpoint path is required.")
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.target_network.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
        self.history_window = int(checkpoint.get("history_window", self.history_window))
        self.action_count = int(checkpoint.get("action_count", self.action_count))


class QLearningAgent:
    """Reproducible tabular Q-learning over a configurable public state.

    ``history_window`` is retained for backwards compatibility. New code can
    pass a ``representation`` directly, keeping the RL update independent of
    how recent observations are represented.
    """

    def __init__(
        self,
        history_window: int = 1,
        learning_rate: float = 0.15,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        seed: int | None = None,
        representation: StateRepresentation | None = None,
        action_count: int = 5,
    ) -> None:
        if int(history_window) < 1:
            raise ValueError("history_window must be at least 1.")
        if not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0.0 <= float(discount) <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 <= float(epsilon_min) <= 1.0:
            raise ValueError("epsilon_min must be in [0, 1].")
        if float(epsilon_min) > float(epsilon):
            raise ValueError("epsilon_min cannot exceed epsilon.")
        if not 0.0 < float(epsilon_decay) <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if int(action_count) < 1:
            raise ValueError("action_count must be positive.")
        self.history_window = int(history_window)
        self.lr = float(learning_rate)
        self.gamma = float(discount)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.action_count = int(action_count)
        self.rng = np.random.default_rng(seed)
        self.representation = representation or (
            CurrentObservationRepresentation()
            if self.history_window == 1
            else HistoryRepresentation(self.history_window)
        )
        self.history_window = int(
            getattr(self.representation, "window_size", self.history_window)
        )
        self.q: defaultdict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_count, dtype=float)
        )

    @property
    def representation_name(self) -> str:
        return self.representation.name

    def reset_episode(self, observation: Observation) -> None:
        self.representation.reset(observation)

    def current_state(self) -> State:
        return self.representation.state()

    def _state(self, observation: Observation) -> State:
        """Return current state, repairing a caller's stale latest observation."""
        if self.representation.latest_token() != observation_token(observation):
            self.representation.update(observation)
        return self.current_state()

    def act(
        self,
        observation: Observation,
        explore: bool = True,
        deterministic_ties: bool = False,
    ) -> tuple[int, State]:
        state = self._state(observation)
        if explore and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, self.action_count))
        else:
            values = self.q.get(state)
            if values is None:
                values = np.zeros(self.action_count, dtype=float)
            choices = np.flatnonzero(values == values.max())
            action = int(choices[0] if deterministic_ties else self.rng.choice(choices))
        return action, state

    def advance_history(self, next_observation: Observation) -> None:
        """Record one next observation during either training or evaluation."""
        self.representation.update(next_observation)

    def update_q(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Apply one Q-learning update without touching sensory context."""
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_count:
            raise ValueError(f"Action {action} is outside the action space.")
        old = self.q[state][int(action)]
        future = 0.0 if done else float(np.max(self.q[next_state]))
        target = float(reward) + self.gamma * future
        self.q[state][int(action)] += self.lr * (target - old)

    def learn(
        self,
        state: State,
        action: int,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        """Backwards-compatible combined history advance and Q update."""
        self.advance_history(next_observation)
        self.update_q(state, action, reward, self.current_state(), done)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def q_table_copy(self) -> dict[State, np.ndarray]:
        """Return a copy for tests verifying evaluation performs no learning."""
        return {state: values.copy() for state, values in self.q.items()}
