from src.environment import OdorSearchEnv
from src.experiment import run_experiment


def test_experiment_run_writes_documented_artifacts(tmp_path):
    config = {
        "grid_size": 7,
        "max_steps": 20,
        "training_episodes": 5,
        "evaluation_episodes": 6,
        "wind_change_step": 8,
        "history_window": 2,
        "seed": 11,
    }
    output_dir = tmp_path / "artifact_run"

    run_experiment(config, output_dir)

    assert (output_dir / "config.json").exists()
    assert (output_dir / "run_log.txt").exists()
    assert (output_dir / "training_metrics.csv").exists()
    assert (output_dir / "evaluation_episodes.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert any(output_dir.glob("*.png"))


def test_reset_returns_valid_observation():
    env = OdorSearchEnv(grid_size=10, seed=1)
    obs = env.reset()
    assert set(obs) == {"odor", "wind"}
    assert obs["odor"] in (0, 1)
    assert obs["wind"] in (0, 1, 2, 3)


def test_step_returns_expected_values():
    env = OdorSearchEnv(grid_size=10, seed=2)
    env.reset()
    obs, reward, done, info = env.step(4)
    assert set(obs) == {"odor", "wind"}
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info.position == env.position


def test_wind_change_occurs_on_configured_step():
    env = OdorSearchEnv(grid_size=10, wind_change_step=2, seed=3)
    env.reset()
    _, _, _, first = env.step(4)
    _, _, _, second = env.step(4)
    assert not first.wind_changed
    assert second.wind_changed
