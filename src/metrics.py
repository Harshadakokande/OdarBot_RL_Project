"""Evaluation-only metrics for the controlled wind-change experiment.

The recovery measure is intentionally operational rather than biological: after
the configured wind event, a recovery is a configurable streak of *agent-
visible* positive odor readings.  It is useful for a transparent preliminary
comparison, but it is not evidence of genuine plume tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RecoveryConfig:
    """Parameters for the explicitly labelled recovery proxy."""

    consecutive_detections: int = 2

    def __post_init__(self) -> None:
        if int(self.consecutive_detections) < 1:
            raise ValueError("consecutive_detections must be at least 1.")


class RecoveryTracker:
    """Track post-event odor reacquisition and sustained-detection proxy.

    The environment applies the wind change after action ``k`` and emits the
    resulting post-change observation on that same transition.  Therefore a
    detection in the event observation has elapsed recovery time zero.  This
    convention is recorded in the documentation and avoids silently skipping a
    post-change sensor sample.
    """

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        self.wind_change_step: int | None = None
        self.first_reacquisition_step: int | None = None
        self.recovery_step: int | None = None
        self._streak = 0
        self._odor_sensor_seen_after_change = False

    def update(
        self,
        *,
        step_count: int,
        wind_changed: bool,
        observation: Mapping[str, int],
    ) -> None:
        """Consume one returned environment observation after a transition."""
        if wind_changed:
            self.wind_change_step = int(step_count)
            self._streak = 0
        if self.wind_change_step is None:
            return
        if "odor" not in observation:
            return

        self._odor_sensor_seen_after_change = True
        odor_detected = int(observation["odor"]) == 1
        if odor_detected:
            if self.first_reacquisition_step is None:
                self.first_reacquisition_step = int(step_count)
            self._streak += 1
            if (
                self.recovery_step is None
                and self._streak >= self.config.consecutive_detections
            ):
                self.recovery_step = int(step_count)
        else:
            self._streak = 0

    def as_record(self) -> dict[str, Any]:
        """Return serializable fields, retaining unrecovered trials explicitly."""
        changed = self.wind_change_step is not None
        available = changed and self._odor_sensor_seen_after_change
        first_time = (
            self.first_reacquisition_step - self.wind_change_step
            if self.first_reacquisition_step is not None and self.wind_change_step is not None
            else None
        )
        recovery_time = (
            self.recovery_step - self.wind_change_step
            if self.recovery_step is not None and self.wind_change_step is not None
            else None
        )
        observed = self.recovery_step is not None
        return {
            "wind_change_observed": changed,
            "wind_change_step": self.wind_change_step,
            "first_odor_reacquisition_step": self.first_reacquisition_step,
            "first_odor_reacquisition_time": first_time,
            "tracking_recovery_step": self.recovery_step,
            "tracking_recovery_time": recovery_time,
            "sustained_odor_detection_proxy_step": self.recovery_step,
            "sustained_odor_detection_proxy_time": recovery_time,
            "recovery_metric_available": available,
            "recovery_observed": observed,
            "recovery_not_observed": bool(available and not observed),
            "wind_change_event_occured": changed,
            "episode_ended_before_wind_change": False,
        }


def path_length(path: Sequence[tuple[int, int]]) -> int:
    """Manhattan path length; stay actions contribute zero spatial distance."""
    return int(
        sum(
            abs(current[0] - previous[0]) + abs(current[1] - previous[1])
            for previous, current in zip(path, path[1:])
        )
    )


def mean_confidence_interval(values: Sequence[float]) -> tuple[float, float, float]:
    """Return mean and approximate 95% normal CI for independent seed means.

    A CI is not meaningful for a single seed, so its bounds are NaN in that
    case.  The result deliberately avoids adding a scientific-computing
    dependency solely for a t-distribution lookup.
    """
    usable = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(usable) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(usable))
    if len(usable) < 2:
        return mean, float("nan"), float("nan")
    margin = 1.96 * float(np.std(usable, ddof=1)) / float(np.sqrt(len(usable)))
    return mean, mean - margin, mean + margin


def _numeric_mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _numeric_median(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.median(values)) if values else float("nan")


def summarize_episode_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate episode records by method and evaluation condition.

    Variability and approximate confidence intervals are calculated over
    independent *training-seed* means, not by incorrectly treating all matched
    evaluation episodes as independent training runs.
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["method"]), str(record["evaluation_condition"]))
        groups.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for (method, condition), group in sorted(groups.items()):
        successes = [float(record["success"]) for record in group]
        successful = [record for record in group if bool(record["success"])]
        wind_change_event_episodes = [
            record for record in group if bool(record.get("wind_change_observed"))
        ]
        ended_before_change = [
            record for record in group if bool(record.get("episode_ended_before_wind_change"))
        ]
        recovery_eligible = [
            record for record in group if bool(record.get("recovery_metric_available"))
        ]
        recovered = [record for record in recovery_eligible if bool(record["recovery_observed"])]

        by_seed: dict[int, list[Mapping[str, Any]]] = {}
        for record in group:
            by_seed.setdefault(int(record["training_seed"]), []).append(record)
        seed_success_rates = [
            float(np.mean([float(row["success"]) for row in seed_rows]))
            for seed_rows in by_seed.values()
        ]
        success_mean, success_low, success_high = mean_confidence_interval(seed_success_rates)

        summaries.append(
            {
                "method": method,
                "evaluation_condition": condition,
                "training_seed_count": len(by_seed),
                "evaluation_episode_count": len(group),
                "success_rate": float(np.mean(successes)) if successes else float("nan"),
                "success_rate_seed_mean": success_mean,
                "success_rate_ci95_low": success_low,
                "success_rate_ci95_high": success_high,
                "success_rate_seed_std": (
                    float(np.std(seed_success_rates, ddof=1))
                    if len(seed_success_rates) > 1
                    else float("nan")
                ),
                "mean_episode_steps": _numeric_mean(group, "episode_steps"),
                "median_episode_steps": _numeric_median(group, "episode_steps"),
                "mean_steps_to_source_success_only": _numeric_mean(successful, "episode_steps"),
                "median_steps_to_source_success_only": _numeric_median(successful, "episode_steps"),
                "mean_path_length": _numeric_mean(group, "path_length"),
                "mean_pre_change_odor_detection_rate": _numeric_mean(
                    group, "pre_change_odor_detection_rate"
                ),
                "mean_post_change_odor_detection_rate": _numeric_mean(
                    group, "post_change_odor_detection_rate"
                ),
                "wind_change_event_episode_count": len(wind_change_event_episodes),
                "episodes_ending_before_wind_change_count": len(ended_before_change),
                "episodes_with_wind_change_or_post_change_observations_count": len(recovery_eligible),
                "recovery_eligible_episode_count": len(recovery_eligible),
                "recovery_observed_episode_count": len(recovered),
                "recovery_not_observed_episode_count": sum(
                    int(bool(record.get("recovery_not_observed"))) for record in group
                ),
                "wind_change_event_fraction": (
                    len(wind_change_event_episodes) / len(group)
                    if group
                    else float("nan")
                ),
                "episodes_ending_before_wind_change_fraction": (
                    len(ended_before_change) / len(group)
                    if group
                    else float("nan")
                ),
                "recovery_observed_fraction": (
                    len(recovered) / len(recovery_eligible)
                    if recovery_eligible
                    else float("nan")
                ),
                "recovery_not_observed_fraction": (
                    sum(int(bool(record.get("recovery_not_observed"))) for record in group)
                    / len(recovery_eligible)
                    if recovery_eligible
                    else float("nan")
                ),
                "mean_tracking_recovery_time_observed_only": _numeric_mean(
                    recovered, "tracking_recovery_time"
                ),
                "median_tracking_recovery_time_observed_only": _numeric_median(
                    recovered, "tracking_recovery_time"
                ),
                "mean_sustained_odor_detection_proxy_time_observed_only": _numeric_mean(
                    recovered, "sustained_odor_detection_proxy_time"
                ),
                "median_sustained_odor_detection_proxy_time_observed_only": _numeric_median(
                    recovered, "sustained_odor_detection_proxy_time"
                ),
            }
        )
    return summaries


def recovery_config_dict(config: RecoveryConfig) -> dict[str, Any]:
    """Stable serializable helper used in run configuration snapshots."""
    return asdict(config)
