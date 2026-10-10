"""Focused behavioral tests for agent representations and recovery metrics."""

import csv

from src.agents import (
    DQNAgent,
    GenericContextRepresentation,
    HistoryRepresentation,
    QLearningAgent,
    observation_token,
)
from src.experiment import run_experiment
from src.metrics import RecoveryConfig, RecoveryTracker, summarize_episode_records


def test_history_representation_updates_and_resets_between_episodes():
    """A new episode must not retain sensory tokens from a previous one."""
    initial = {"odor": 0, "wind": 1}
    second = {"odor": 1, "wind": 1}
    third = {"odor": 0, "wind": 2}
    fresh_episode = {"odor": 1, "wind": 3}
    agent = QLearningAgent(
        representation=HistoryRepresentation(window_size=3), action_count=5, seed=2
    )

    agent.reset_episode(initial)
    initial_state = agent.current_state()
    assert len(initial_state) == 3
    assert initial_state[-1] == observation_token(initial)
    assert initial_state[:2] == (initial_state[0], initial_state[0])

    agent.advance_history(second)
    assert agent.current_state()[-2:] == (
        observation_token(initial),
        observation_token(second),
    )
    agent.advance_history(third)
    assert agent.current_state() == (
        observation_token(initial),
        observation_token(second),
        observation_token(third),
    )

    agent.reset_episode(fresh_episode)
    reset_state = agent.current_state()
    assert reset_state[-1] == observation_token(fresh_episode)
    assert reset_state[:2] == (initial_state[0], initial_state[0])
    assert observation_token(third) not in reset_state


def test_generic_context_representation_is_explicit_placeholder_and_summarizes_context():
    """The optional context baseline must be usable while not being labelled ACIL."""
    representation = GenericContextRepresentation(window_size=3)
    first = {"odor": 0, "wind": 0, "previous_action": 4}
    second = {"odor": 1, "wind": 0, "previous_action": 0}
    latest = {"odor": 1, "wind": 1, "previous_action": 0}

    representation.reset(first)
    representation.update(second)
    representation.update(latest)

    assert representation.name == "generic_context_placeholder_not_acil"
    assert representation.state() == (
        "generic_context",
        observation_token(latest),
        2,  # positive odor readings in the retained window
        1,  # one wind-direction change
        1,  # one previous-action change
    )


def test_dqn_history_does_not_duplicate_the_current_observation():
    """The DQN agent should only append the next observed state after an action."""
    agent = DQNAgent(action_count=5, history_window=2, seed=9)
    obs = {"odor": 1, "wind": 0}

    agent.reset_episode(obs)
    assert len(agent.history) == 1
    agent.act(obs, explore=False)
    assert len(agent.history) == 1

    next_obs = {"odor": 0, "wind": 1}
    agent.advance_history(next_obs)
    assert len(agent.history) == 2
    assert agent.current_state_vector().shape == (2 * 3,)


def test_recovery_tracker_records_first_reacquisition_and_consecutive_recovery():
    """A post-change detection at the event has elapsed time zero."""
    tracker = RecoveryTracker(RecoveryConfig(consecutive_detections=2))

    tracker.update(step_count=4, wind_changed=False, observation={"odor": 1})
    tracker.update(step_count=5, wind_changed=True, observation={"odor": 1})
    tracker.update(step_count=6, wind_changed=False, observation={"odor": 1})
    record = tracker.as_record()

    assert record["wind_change_step"] == 5
    assert record["first_odor_reacquisition_step"] == 5
    assert record["first_odor_reacquisition_time"] == 0
    assert record["tracking_recovery_step"] == 6
    assert record["tracking_recovery_time"] == 1
    assert record["recovery_observed"] is True
    assert record["recovery_not_observed"] is False


def test_recovery_tracker_requires_consecutive_detections():
    """An intervening negative odor reading restarts the tracking proxy streak."""
    tracker = RecoveryTracker(RecoveryConfig(consecutive_detections=2))

    tracker.update(step_count=10, wind_changed=True, observation={"odor": 1})
    tracker.update(step_count=11, wind_changed=False, observation={"odor": 0})
    tracker.update(step_count=12, wind_changed=False, observation={"odor": 1})
    tracker.update(step_count=13, wind_changed=False, observation={"odor": 1})
    record = tracker.as_record()

    assert record["first_odor_reacquisition_step"] == 10
    assert record["tracking_recovery_step"] == 13
    assert record["tracking_recovery_time"] == 3


def test_recovery_tracker_retains_no_recovery_case_explicitly():
    """Failed recovery trials remain visible instead of being dropped from data."""
    tracker = RecoveryTracker(RecoveryConfig(consecutive_detections=2))

    tracker.update(step_count=3, wind_changed=True, observation={"odor": 0})
    tracker.update(step_count=4, wind_changed=False, observation={"odor": 1})
    tracker.update(step_count=5, wind_changed=False, observation={"odor": 0})
    record = tracker.as_record()

    assert record["recovery_metric_available"] is True
    assert record["first_odor_reacquisition_time"] == 1
    assert record["tracking_recovery_time"] is None
    assert record["recovery_observed"] is False
    assert record["recovery_not_observed"] is True


def test_experiment_records_independent_training_and_evaluation_seeds(tmp_path):
    """Each evaluation episode should keep both the trained model seed and the environment seed."""
    config = {
        "grid_size": 7,
        "max_steps": 20,
        "training_episodes": 5,
        "evaluation_episodes": 4,
        "wind_change_step": 8,
        "history_window": 2,
        "training_seeds": [11, 12],
        "evaluation_seeds": [101, 102, 103, 104],
        "seed": 11,
    }
    output_dir = tmp_path / "seeded_run"

    run_experiment(config, output_dir)

    with (output_dir / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {int(row["training_seed"]) for row in rows} == {11, 12}
    assert {int(row["evaluation_seed"]) for row in rows} == {101, 102, 103, 104}
    assert len(rows) == 16


def test_experiment_results_csv_is_non_empty_after_full_run(tmp_path):
    """The exported summary file must remain populated and not be overwritten with an empty string."""
    config = {
        "grid_size": 7,
        "max_steps": 20,
        "training_episodes": 5,
        "evaluation_episodes": 4,
        "wind_change_step": 8,
        "history_window": 2,
        "training_seeds": [11],
        "evaluation_seeds": [101, 102, 103, 104],
        "seed": 11,
    }
    output_dir = tmp_path / "artifact_run"

    run_experiment(config, output_dir)

    results_path = output_dir / "results.csv"
    assert results_path.exists()
    text = results_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 0
    assert "method" in text
    assert "training_seed" in text


def test_summarize_episode_records_tracks_seed_aggregation_correctly():
    """Seed-level means should be estimated from independent model seeds, not over-aggregated episodes."""
    records = [
        {"method": "Current observation", "evaluation_condition": "wind_change", "success": 1, "training_seed": 1, "episode_steps": 10, "path_length": 9, "recovery_metric_available": True, "recovery_observed": True, "recovery_not_observed": False, "wind_change_observed": True, "episode_ended_before_wind_change": False, "tracking_recovery_time": 2.0, "sustained_odor_detection_proxy_time": 2.0},
        {"method": "Current observation", "evaluation_condition": "wind_change", "success": 0, "training_seed": 1, "episode_steps": 12, "path_length": 11, "recovery_metric_available": True, "recovery_observed": False, "recovery_not_observed": True, "wind_change_observed": True, "episode_ended_before_wind_change": False, "tracking_recovery_time": None, "sustained_odor_detection_proxy_time": None},
        {"method": "Current observation", "evaluation_condition": "wind_change", "success": 1, "training_seed": 2, "episode_steps": 11, "path_length": 10, "recovery_metric_available": True, "recovery_observed": True, "recovery_not_observed": False, "wind_change_observed": True, "episode_ended_before_wind_change": False, "tracking_recovery_time": 3.0, "sustained_odor_detection_proxy_time": 3.0},
    ]
    summary = summarize_episode_records(records)
    assert summary[0]["training_seed_count"] == 2
    assert summary[0]["recovery_observed_episode_count"] == 2
    assert summary[0]["recovery_not_observed_episode_count"] == 1
