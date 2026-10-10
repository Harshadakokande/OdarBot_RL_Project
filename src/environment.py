"""Configurable, reproducible 2D odor-source-search environment.

The odor field in this module is deliberately a *heuristic detection field*,
not a fluid-dynamics or biological plume simulator. Positions are ``(x, y)``:
``x`` increases to the east (right) and ``y`` increases to the south (down).
Wind codes describe the direction air travels: ``0=N, 1=E, 2=S, 3=W``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np


Position = tuple[int, int]


@dataclass(frozen=True)
class EnvConfig:
    """Parameters for :class:`OdorSearchEnv`.

    ``odor_intermittency`` is the probability that the latent odor signal is
    available for an individual observation. ``false_positive_rate`` and
    ``false_negative_rate`` model asymmetric binary-sensor errors, while
    ``sensor_noise_probability`` is an optional final bit flip. The defaults
    preserve the project's original binary, noisy sensor behaviour closely.

    A fixed source, start, initial wind, or sensor seed is useful for a
    controlled experiment. They can also be supplied per episode with an
    :class:`EpisodeScenario`, which takes precedence during ``reset``.
    """

    grid_size: int = 15
    max_steps: int = 160
    wind_change_step: Optional[int] = 55
    change_wind: bool = True
    wind_change_delta: int = 1

    source_position: Optional[Position] = None
    start_position: Optional[Position] = None
    initial_wind: Optional[int] = None
    sensor_seed: Optional[int] = None

    expose_odor: bool = True
    expose_wind: bool = True
    expose_previous_action: bool = False

    odor_baseline: float = 0.04
    odor_strength: float = 0.85
    downwind_scale: float = 2.0
    crosswind_scale: float = 2.5
    distance_scale: float = 10.0
    odor_intermittency: float = 1.0
    false_positive_rate: float = 0.025
    false_negative_rate: float = 0.0
    sensor_noise_probability: float = 0.0

    source_radius: float = 0.0
    step_penalty: float = -0.05
    source_reward: float = 10.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_int("grid_size", self.grid_size, minimum=2)
        _validate_int("max_steps", self.max_steps, minimum=1)
        _validate_bool("change_wind", self.change_wind)
        _validate_bool("expose_odor", self.expose_odor)
        _validate_bool("expose_wind", self.expose_wind)
        _validate_bool("expose_previous_action", self.expose_previous_action)

        if not self.expose_odor and not self.expose_wind:
            raise ValueError(
                "At least one sensor must be exposed: set expose_odor or "
                "expose_wind to True."
            )

        if self.change_wind and self.wind_change_step is not None:
            _validate_int("wind_change_step", self.wind_change_step, minimum=1)
            if self.wind_change_step > self.max_steps:
                raise ValueError("wind_change_step must not exceed max_steps.")
            _validate_wind_delta(self.wind_change_delta)
        elif self.wind_change_step is not None:
            _validate_int("wind_change_step", self.wind_change_step, minimum=1)

        if self.initial_wind is not None:
            _validate_wind("initial_wind", self.initial_wind)
        if self.sensor_seed is not None:
            _validate_seed("sensor_seed", self.sensor_seed)
        if self.seed is not None:
            _validate_seed("seed", self.seed)

        for name, value in (
            ("odor_baseline", self.odor_baseline),
            ("odor_intermittency", self.odor_intermittency),
            ("false_positive_rate", self.false_positive_rate),
            ("false_negative_rate", self.false_negative_rate),
            ("sensor_noise_probability", self.sensor_noise_probability),
        ):
            _validate_probability(name, value)
        for name, value in (
            ("odor_strength", self.odor_strength),
            ("source_radius", self.source_radius),
        ):
            _validate_real(name, value, minimum=0.0)
        for name, value in (
            ("downwind_scale", self.downwind_scale),
            ("crosswind_scale", self.crosswind_scale),
            ("distance_scale", self.distance_scale),
        ):
            _validate_real(name, value, minimum=np.finfo(float).eps)
        _validate_real("step_penalty", self.step_penalty)
        _validate_real("source_reward", self.source_reward)

        for name, position in (
            ("source_position", self.source_position),
            ("start_position", self.start_position),
        ):
            if position is not None:
                _validate_position(name, position, self.grid_size)


@dataclass(frozen=True)
class EpisodeScenario:
    """A fully specified spatial/sensor scenario for one episode.

    Supplying the same scenario and action sequence produces identical sensor
    readings, independently of the environment's layout RNG state.
    """

    source: Position
    start: Position
    initial_wind: int
    sensor_seed: Optional[int] = None


@dataclass(frozen=True)
class StepInfo:
    """Transition diagnostics; this object deliberately contains no source."""

    odor: Optional[int]
    wind: Optional[int]
    found_source: bool
    wind_changed: bool
    position: Position
    step_count: int
    terminated: bool
    truncated: bool


class OdorSearchEnv:
    """Simplified 2D odor search with a hidden source and discrete actions.

    The default constructor remains compatible with the original project::

        OdorSearchEnv(grid_size=15, max_steps=160, wind_change_step=55,
                      change_wind=True, seed=7)

    For new code, pass an ``EnvConfig`` with ``config=``. At a configured,
    one-indexed wind-change step, the agent first moves, the wind then changes,
    and the returned observation is sampled under the *new* wind. Thus the
    observation returned by transition ``k`` is the first post-change
    observation when ``wind_change_step == k``.
    """

    # x increases right/east and y increases down/south.
    ACTIONS = {
        0: (0, -1),  # north / up
        1: (1, 0),   # east / right
        2: (0, 1),   # south / down
        3: (-1, 0),  # west / left
        4: (0, 0),   # stay/sample
    }
    WIND_VECTORS = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
    }

    def __init__(
        self,
        grid_size: int = 15,
        max_steps: int = 160,
        wind_change_step: Optional[int] = 55,
        change_wind: bool = True,
        seed: Optional[int] = None,
        *,
        config: Optional[EnvConfig] = None,
    ) -> None:
        """Create the environment.

        When ``config`` is supplied it takes precedence over legacy positional
        parameters. The ``seed`` keyword still overrides ``config.seed`` so
        existing experiment entry points can control the run seed.
        """

        if config is None:
            config = EnvConfig(
                grid_size=grid_size,
                max_steps=max_steps,
                wind_change_step=wind_change_step,
                change_wind=change_wind,
                seed=seed,
            )
        elif not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig instance.")

        self.config = config
        self.grid_size = config.grid_size
        self.max_steps = config.max_steps
        self.wind_change_step = (
            config.wind_change_step if config.change_wind else None
        )
        self.change_wind = config.change_wind
        self.wind_change_delta = config.wind_change_delta

        effective_seed = seed if seed is not None else config.seed
        if effective_seed is not None:
            _validate_seed("seed", effective_seed)
        self._seed = effective_seed
        self.rng = np.random.default_rng(effective_seed)
        self._sensor_rng = np.random.default_rng()
        self.sensor_seed: Optional[int] = None

        # These are initialized by reset(), then retained for compatibility
        # with the original experiment module.
        self.source: Position
        self.position: Position
        self.initial_position: Position
        self.initial_wind: int
        self.wind: int
        self.last_odor: int
        self.last_action: Optional[int]
        self.step_count: int
        self.wind_changed_at: Optional[int]
        self.done: bool
        self.path: list[Position]
        self.scenario: EpisodeScenario
        self.reset()

    @property
    def current_observation(self) -> dict[str, Optional[int]]:
        """Return the currently available, non-privileged observation."""

        return self.observe()

    def reset(
        self,
        seed: Optional[int] = None,
        scenario: Optional[EpisodeScenario] = None,
    ) -> dict[str, Optional[int]]:
        """Start a new episode and return its first permitted observation.

        ``seed`` resets only the layout-seed stream. The sensor stream is
        derived from that stream unless a scenario or configuration explicitly
        supplies ``sensor_seed``. This separation means actions in one episode
        never perturb layouts sampled for later episodes.
        """

        if seed is not None:
            _validate_seed("seed", seed)
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)

        if scenario is not None and not isinstance(scenario, EpisodeScenario):
            raise TypeError("scenario must be an EpisodeScenario instance.")

        if scenario is None:
            source, start, initial_wind, sensor_seed = self._sample_scenario()
        else:
            source, start, initial_wind, sensor_seed = self._validate_scenario(
                scenario
            )

        self.source = source
        self.position = start
        self.initial_position = start
        self.initial_wind = initial_wind
        self.wind = initial_wind
        self.sensor_seed = sensor_seed
        self._sensor_rng = np.random.default_rng(sensor_seed)
        self.scenario = EpisodeScenario(
            source=source,
            start=start,
            initial_wind=initial_wind,
            sensor_seed=sensor_seed,
        )

        self.step_count = 0
        self.wind_changed_at = None
        self.done = False
        self.path = [self.position]
        self.last_action = None
        self.last_odor = self._odor_reading()
        return self.observe()

    def observe(self) -> dict[str, Optional[int]]:
        """Return only sensor values enabled by ``EnvConfig``.

        The true source and absolute position are intentionally absent. The
        returned dictionary is freshly allocated so caller mutation cannot
        modify environment state.
        """

        observation: dict[str, Optional[int]] = {}
        if self.config.expose_odor:
            observation["odor"] = int(self.last_odor)
        if self.config.expose_wind:
            observation["wind"] = int(self.wind)
        if self.config.expose_previous_action:
            observation["previous_action"] = self.last_action
        return observation

    def _sample_scenario(self) -> tuple[Position, Position, int, int]:
        source = (
            self.config.source_position
            if self.config.source_position is not None
            else self._sample_position()
        )
        start = (
            self.config.start_position
            if self.config.start_position is not None
            else self._sample_start_not_at_source(source)
        )
        initial_wind = (
            self.config.initial_wind
            if self.config.initial_wind is not None
            else int(self.rng.integers(0, 4))
        )
        sensor_seed = (
            self.config.sensor_seed
            if self.config.sensor_seed is not None
            else self._next_sensor_seed()
        )
        self._validate_episode_values(source, start, initial_wind, sensor_seed)
        return source, start, initial_wind, sensor_seed

    def _validate_scenario(
        self, scenario: EpisodeScenario
    ) -> tuple[Position, Position, int, int]:
        sensor_seed = (
            scenario.sensor_seed
            if scenario.sensor_seed is not None
            else self._next_sensor_seed()
        )
        self._validate_episode_values(
            scenario.source, scenario.start, scenario.initial_wind, sensor_seed
        )
        return (
            _as_position(scenario.source),
            _as_position(scenario.start),
            int(scenario.initial_wind),
            int(sensor_seed),
        )

    def _validate_episode_values(
        self, source: Position, start: Position, initial_wind: int, sensor_seed: int
    ) -> None:
        _validate_position("source", source, self.grid_size)
        _validate_position("start", start, self.grid_size)
        _validate_wind("initial_wind", initial_wind)
        _validate_seed("sensor_seed", sensor_seed)
        if self._source_reached(_as_position(start), source=_as_position(source)):
            raise ValueError(
                "start must be outside the configured source-reached radius."
            )

    def _sample_position(self) -> Position:
        return (
            int(self.rng.integers(0, self.grid_size)),
            int(self.rng.integers(0, self.grid_size)),
        )

    def _sample_start_not_at_source(self, source: Position) -> Position:
        valid_positions = [
            (x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if not self._source_reached((x, y), source=source)
        ]
        if not valid_positions:
            raise ValueError(
                "source_radius leaves no valid start position in the grid."
            )
        return valid_positions[int(self.rng.integers(0, len(valid_positions)))]

    def _next_sensor_seed(self) -> int:
        # Draw from layout RNG once per episode, not per sensor read. This
        # keeps later layouts independent of episode length and actions.
        return int(self.rng.integers(0, np.iinfo(np.int64).max))

    def _source_reached(
        self, position: Position, *, source: Optional[Position] = None
    ) -> bool:
        sx, sy = self.source if source is None else source
        px, py = position
        dx, dy = px - sx, py - sy
        return dx * dx + dy * dy <= self.config.source_radius ** 2

    def _odor_probability(
        self, position: Position, wind: Optional[int] = None
    ) -> float:
        """Return latent binary-detection probability before sensor effects."""

        _validate_position("position", position, self.grid_size)
        active_wind = self.wind if wind is None else wind
        _validate_wind("wind", active_wind)
        px, py = position
        sx, sy = self.source
        dx, dy = px - sx, py - sy
        distance = float(np.hypot(dx, dy))

        # The plume is strongest downwind, narrows crosswind, and decays with
        # distance. It is a transparent heuristic, not a validated model.
        wx, wy = self.WIND_VECTORS[int(active_wind)]
        downwind = dx * wx + dy * wy
        crosswind = abs(dx * wy - dy * wx)
        along = 1.0 / (
            1.0 + np.exp(-np.clip(downwind / self.config.downwind_scale, -20, 20))
        )
        cross = np.exp(-crosswind / self.config.crosswind_scale)
        distance_factor = np.exp(-distance / self.config.distance_scale)
        probability = self.config.odor_baseline + (
            self.config.odor_strength * along * cross * distance_factor
        )
        return float(np.clip(probability, 0.0, 1.0))

    def _odor_reading(self) -> int:
        """Sample one binary measurement with fixed draws per observation."""

        # Always consume four values. It makes sensor traces stable for a
        # fixed scenario/action sequence even when a branch changes.
        signal_available_draw = float(self._sensor_rng.random())
        detection_draw = float(self._sensor_rng.random())
        error_draw = float(self._sensor_rng.random())
        noise_draw = float(self._sensor_rng.random())

        signal_available = signal_available_draw < self.config.odor_intermittency
        latent_detected = signal_available and (
            detection_draw < self._odor_probability(self.position)
        )
        if latent_detected:
            measured = error_draw >= self.config.false_negative_rate
        else:
            measured = error_draw < self.config.false_positive_rate
        if noise_draw < self.config.sensor_noise_probability:
            measured = not measured
        return int(measured)

    def step(
        self, action: int
    ) -> tuple[dict[str, Optional[int]], float, bool, StepInfo]:
        """Apply one discrete action and return observation, reward, done, info."""

        if self.done:
            raise RuntimeError("Episode finished; call reset() before stepping.")
        _validate_action(action)
        action = int(action)

        self.step_count += 1
        dx, dy = self.ACTIONS[action]
        x, y = self.position
        self.position = (
            int(np.clip(x + dx, 0, self.grid_size - 1)),
            int(np.clip(y + dy, 0, self.grid_size - 1)),
        )
        self.path.append(self.position)
        self.last_action = action

        wind_changed = False
        if (
            self.change_wind
            and self.wind_change_step is not None
            and self.step_count == self.wind_change_step
        ):
            self.wind = (self.wind + self.wind_change_delta) % 4
            self.wind_changed_at = self.step_count
            wind_changed = True

        found_source = self._source_reached(self.position)
        terminated = found_source
        truncated = not terminated and self.step_count >= self.max_steps
        self.done = terminated or truncated
        self.last_odor = self._odor_reading()
        observation = self.observe()
        reward = self.config.source_reward if found_source else self.config.step_penalty
        info = StepInfo(
            odor=int(self.last_odor) if self.config.expose_odor else None,
            wind=int(self.wind) if self.config.expose_wind else None,
            found_source=found_source,
            wind_changed=wind_changed,
            position=self.position,
            step_count=self.step_count,
            terminated=terminated,
            truncated=truncated,
        )
        return observation, float(reward), self.done, info

    def render_odor_field(self, wind: Optional[int] = None) -> np.ndarray:
        """Return the deterministic latent field for the current or given wind."""

        if wind is not None:
            _validate_wind("wind", wind)
        field = np.zeros((self.grid_size, self.grid_size), dtype=float)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                field[y, x] = self._odor_probability((x, y), wind=wind)
        return field


def _validate_int(name: str, value: object, *, minimum: Optional[int] = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if minimum is not None and int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _validate_real(
    name: str, value: object, *, minimum: Optional[float] = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise TypeError(f"{name} must be a finite real number.")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _validate_probability(name: str, value: object) -> None:
    _validate_real(name, value, minimum=0.0)
    if float(value) > 1.0:
        raise ValueError(f"{name} must be at most 1.0.")


def _validate_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_wind(name: str, value: object) -> None:
    _validate_int(name, value, minimum=0)
    if int(value) > 3:
        raise ValueError(f"{name} must be one of 0 (N), 1 (E), 2 (S), or 3 (W).")


def _validate_wind_delta(value: object) -> None:
    _validate_int("wind_change_delta", value, minimum=1)
    if int(value) > 3:
        raise ValueError("wind_change_delta must be between 1 and 3.")


def _validate_seed(name: str, value: object) -> None:
    _validate_int(name, value, minimum=0)


def _as_position(position: Position) -> Position:
    return (int(position[0]), int(position[1]))


def _validate_position(name: str, position: object, grid_size: int) -> None:
    if not isinstance(position, tuple) or len(position) != 2:
        raise TypeError(f"{name} must be an (x, y) tuple.")
    x, y = position
    _validate_int(f"{name}[0]", x)
    _validate_int(f"{name}[1]", y)
    if not (0 <= int(x) < grid_size and 0 <= int(y) < grid_size):
        raise ValueError(
            f"{name} must lie within grid coordinates 0 through {grid_size - 1}."
        )


def _validate_action(action: object) -> None:
    _validate_int("action", action)
    if int(action) not in OdorSearchEnv.ACTIONS:
        raise ValueError("action must be one of 0, 1, 2, 3, or 4.")
