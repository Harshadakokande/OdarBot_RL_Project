import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents import DQNAgent
from src.environment import OdorSearchEnv
from src.metrics import RecoveryConfig, RecoveryTracker, path_length, summarize_episode_records


def _seed_list(values, *, default_seed: int, default_count: int) -> list[int]:
    if values is None:
        return [int(default_seed) + idx for idx in range(max(1, int(default_count)))]
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    if isinstance(values, (list, tuple, set)):
        seeds = [int(value) for value in values]
        if not seeds:
            raise ValueError("Seed list must not be empty.")
        return seeds
    raise TypeError("Seed configuration must be an int or a list of ints.")


def _method_specs(config):
    history_window = int(config.get("history_window", 1))
    return [
        ("Current observation", 1),
        (f"History ({history_window} steps)", history_window),
    ]


def train_agent(config, history_window, seed):
    env = OdorSearchEnv(
        grid_size=config["grid_size"],
        max_steps=config["max_steps"],
        wind_change_step=config["wind_change_step"],
        change_wind=config.get("change_wind", True),
        seed=seed,
    )
    agent = DQNAgent(
        action_count=5,
        history_window=history_window,
        learning_rate=config.get("learning_rate", 1e-3),
        gamma=config.get("gamma", 0.99),
        epsilon=config.get("epsilon", 1.0),
        epsilon_min=config.get("epsilon_min", 0.05),
        epsilon_decay=config.get("epsilon_decay", 0.995),
        replay_capacity=config.get("replay_capacity", 5000),
        batch_size=config.get("batch_size", 64),
        target_update_interval=config.get("target_update_interval", 200),
        seed=seed + 1000,
    )
    training_rows = []
    for episode_index in range(config["training_episodes"]):
        obs = env.reset()
        agent.reset_episode(obs)
        done = False
        episode_reward = 0.0
        while not done:
            state = agent.current_state_vector()
            action = agent.act(obs, explore=True)
            next_obs, reward, done, _ = env.step(action)
            agent.advance_history(next_obs)
            next_state = agent.current_state_vector()
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            obs = next_obs
            episode_reward += reward
        training_rows.append(
            {
                "method": agent.representation_name,
                "training_seed": seed,
                "episode_index": episode_index,
                "episode_reward": episode_reward,
                "episode_steps": env.step_count,
                "loss": loss if loss is not None else float("nan"),
            }
        )
    return agent, training_rows


def evaluate_agent(config, agent, method_name, training_seed, evaluation_seed_list, keep_path=False):
    records = []
    sample = None
    for episode_index, evaluation_seed in enumerate(evaluation_seed_list):
        env = OdorSearchEnv(
            grid_size=config["grid_size"],
            max_steps=config["max_steps"],
            wind_change_step=config["wind_change_step"],
            change_wind=config.get("change_wind", True),
            seed=evaluation_seed,
        )
        obs = env.reset()
        agent.reset_episode(obs)
        done = False
        tracker = RecoveryTracker(
            RecoveryConfig(consecutive_detections=config.get("consecutive_detections", 2))
        )
        pre_change_hits = 0
        post_change_hits = 0
        pre_change_total = 0
        post_change_total = 0
        while not done:
            action = agent.act(obs, explore=False)
            next_obs, _, done, info = env.step(action)
            agent.advance_history(next_obs)
            current_step = env.step_count
            tracker.update(
                step_count=current_step,
                wind_changed=info.wind_changed,
                observation=next_obs,
            )
            if env.wind_change_step is None or current_step < env.wind_change_step:
                pre_change_total += 1
                pre_change_hits += int(next_obs.get("odor", 0) == 1)
            elif current_step >= env.wind_change_step:
                post_change_total += 1
                post_change_hits += int(next_obs.get("odor", 0) == 1)
            obs = next_obs

        metric_record = tracker.as_record()
        record = {
            "method": method_name,
            "training_seed": int(training_seed),
            "evaluation_seed": int(evaluation_seed),
            "evaluation_condition": "wind_change" if config.get("change_wind", True) else "no_wind_change",
            "episode_index": episode_index,
            "success": int(env.position == env.source),
            "episode_steps": env.step_count,
            "path_length": path_length(env.path),
            "wind_change_observed": bool(metric_record["wind_change_observed"]),
            "wind_change_step": metric_record["wind_change_step"],
            "episode_ended_before_wind_change": bool(
                env.wind_change_step is not None and env.step_count < env.wind_change_step
            ),
            "first_odor_reacquisition_step": metric_record["first_odor_reacquisition_step"],
            "first_odor_reacquisition_time": metric_record["first_odor_reacquisition_time"],
            "tracking_recovery_step": metric_record["tracking_recovery_step"],
            "tracking_recovery_time": metric_record["tracking_recovery_time"],
            "sustained_odor_detection_proxy_step": metric_record["tracking_recovery_step"],
            "sustained_odor_detection_proxy_time": metric_record["tracking_recovery_time"],
            "recovery_metric_available": metric_record["recovery_metric_available"],
            "recovery_observed": metric_record["recovery_observed"],
            "recovery_not_observed": metric_record["recovery_not_observed"],
            "pre_change_odor_detection_rate": (pre_change_hits / pre_change_total) if pre_change_total else float("nan"),
            "post_change_odor_detection_rate": (post_change_hits / post_change_total) if post_change_total else float("nan"),
        }
        records.append(record)
        if keep_path and episode_index == 0:
            sample = {"path": list(env.path), "source": env.source, "start": env.initial_position}

    result = {
        "success_rate": float(np.mean([record["success"] for record in records])) if records else float("nan"),
        "mean_steps": float(np.mean([record["episode_steps"] for record in records])) if records else float("nan"),
        "mean_distance": float(np.mean([record["path_length"] for record in records])) if records else float("nan"),
        "mean_recovery_steps": float(np.mean([record["tracking_recovery_time"] for record in records if record["tracking_recovery_time"] is not None])) if any(record["tracking_recovery_time"] is not None for record in records) else float("nan"),
        "recovery_observed_fraction": float(sum(int(record["recovery_observed"]) for record in records) / len(records)) if records else float("nan"),
        "wind_change_event_fraction": float(sum(int(record["wind_change_observed"]) for record in records) / len(records)) if records else float("nan"),
        "episodes_ended_before_wind_change": sum(int(record["episode_ended_before_wind_change"]) for record in records),
    }
    return records, sample, result


def save_plots(config, metrics, samples, training_rows, outdir):
    env = OdorSearchEnv(grid_size=config["grid_size"], seed=config["seed"])
    field = env.render_odor_field()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(field, origin="upper", cmap="viridis")
    ax.scatter([env.source[0]], [env.source[1]], marker="*", s=220, label="Hidden source")
    ax.set_title("Simplified odor-detection field and source")
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Detection probability")
    fig.tight_layout()
    fig.savefig(outdir / "odor_field.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, sample in samples.items():
        if sample:
            xs = [p[0] for p in sample["path"]]
            ys = [p[1] for p in sample["path"]]
            ax.plot(xs, ys, marker=".", markersize=2, linewidth=1, label=name)
            ax.scatter([sample["source"][0]], [sample["source"][1]], marker="*", s=200)
            ax.scatter([sample["start"][0]], [sample["start"][1]], marker="x", s=70)
    ax.set_xlim(-0.5, config["grid_size"] - 0.5)
    ax.set_ylim(config["grid_size"] - 0.5, -0.5)
    ax.set_xticks(range(config["grid_size"]))
    ax.set_yticks(range(config["grid_size"]))
    ax.grid(alpha=0.2)
    ax.set_title("Illustrative evaluation trajectories")
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "sample_trajectories.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(metrics)
    success_rates = [metrics[name]["success_rate"] for name in names]
    ax.bar(names, [value * 100.0 for value in success_rates])
    ax.set_title("Source-finding success rate by method")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(outdir / "success_rate_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = [metrics[name]["mean_recovery_steps"] for name in names]
    ax.bar(names, means, color=["tab:blue", "tab:orange"][: len(names)])
    ax.set_title("Mean sustained-odor-detection proxy time after wind change")
    ax.set_ylabel("Steps after change")
    fig.tight_layout()
    fig.savefig(outdir / "recovery_comparison.png", dpi=160)
    plt.close(fig)

    if training_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        by_method = {}
        for row in training_rows:
            by_method.setdefault(row["method"], []).append(float(row["episode_reward"]))
        for label, values in by_method.items():
            ax.plot(range(len(values)), values, label=label)
        ax.set_title("Training progress")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode reward")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "learning_progress.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if metrics:
        x = np.arange(len(names))
        pre = [0.0 for _ in names]
        post = [0.0 for _ in names]
        for idx, name in enumerate(names):
            pre[idx] = metrics[name]["success_rate"]
            post[idx] = metrics[name]["mean_recovery_steps"] if np.isfinite(metrics[name]["mean_recovery_steps"]) else 0.0
        ax.bar(x - 0.2, pre, width=0.35, label="Success rate")
        ax.bar(x + 0.2, post, width=0.35, label="Proxy recovery time")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=12)
        ax.set_title("Comparison summary")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if config.get("wind_change_step") is not None:
        ax.axvline(config["wind_change_step"], color="red", linestyle="--", label="Wind change step")
    ax.set_title("Wind-change event timing")
    ax.set_xlabel("Episode step")
    ax.set_ylabel("Reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "wind_change_timeline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if metrics:
        for name in names:
            value = metrics[name]["recovery_observed_fraction"]
            ax.bar(name, value, label=name)
        ax.set_title("Observed proxy-recovery fraction")
        ax.set_ylabel("Fraction of eligible episodes")
        ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(outdir / "pre_post_comparison.png", dpi=160)
    plt.close(fig)


def run_experiment(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = dict(config)
    config.setdefault("change_wind", True)
    config.setdefault("history_window", 1)
    config.setdefault("consecutive_detections", 2)
    config.setdefault("training_episodes", 50)
    config.setdefault("evaluation_episodes", 20)
    config.setdefault("seed", 0)

    training_seeds = _seed_list(
        config.get("training_seeds"),
        default_seed=int(config["seed"]),
        default_count=1,
    )
    evaluation_seeds = _seed_list(
        config.get("evaluation_seeds"),
        default_seed=int(config["seed"]) + 500,
        default_count=int(config["evaluation_episodes"]),
    )
    if len(evaluation_seeds) < int(config["evaluation_episodes"]):
        evaluation_seeds = list(evaluation_seeds) * (int(config["evaluation_episodes"]) // len(evaluation_seeds) + 1)
        evaluation_seeds = evaluation_seeds[: int(config["evaluation_episodes"])]

    run_id = config.get("run_id") or output_dir.name
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    log_lines = [
        f"OdorBot run_id={run_id}",
        "Caution: simplified educational environment, not a fluid-dynamics simulator.",
        f"Training seeds: {training_seeds}",
        f"Evaluation seeds: {evaluation_seeds}",
    ]
    with (output_dir / "run_log.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")

    methods = _method_specs(config)
    all_training_rows = []
    evaluation_records = []
    method_training_rows = {}
    samples = {}

    print("OdorBot: training independent DQN models for the current-observation and sensory-history conditions.")
    print("Caution: simplified educational environment, not a physical plume model.")

    for training_seed in training_seeds:
        for method_name, window in methods:
            print(f"\nTraining {method_name} with seed {training_seed}...")
            agent, training_rows = train_agent(config, window, training_seed)
            all_training_rows.extend(training_rows)
            print(f"Evaluating {method_name} with matched evaluation seeds...")
            records, sample, result = evaluate_agent(
                config,
                agent,
                method_name,
                training_seed,
                evaluation_seeds[: int(config["evaluation_episodes"])],
                keep_path=(training_seed == training_seeds[0] and method_name == methods[0][0]),
            )
            evaluation_records.extend(records)
            method_training_rows.setdefault(method_name, []).append(result)
            samples.setdefault(method_name, sample)
            print(f"  Success rate: {result['success_rate']:.1%}")
            print(f"  Mean steps: {result['mean_steps']:.1f}")
            print(f"  Mean distance: {result['mean_distance']:.1f}")
            print(f"  Mean sustained-odor-detection proxy: {result['mean_recovery_steps']:.1f}; observed in {result['recovery_observed_fraction']:.1%} of eligible episodes")

    metrics = {}
    for method_name, rows in method_training_rows.items():
        metrics[method_name] = {
            "success_rate": float(np.mean([row["success_rate"] for row in rows])) if rows else float("nan"),
            "mean_steps": float(np.mean([row["mean_steps"] for row in rows])) if rows else float("nan"),
            "mean_distance": float(np.mean([row["mean_distance"] for row in rows])) if rows else float("nan"),
            "mean_recovery_steps": float(np.mean([row["mean_recovery_steps"] for row in rows if np.isfinite(row["mean_recovery_steps"])])) if any(np.isfinite(row["mean_recovery_steps"]) for row in rows) else float("nan"),
            "recovery_observed_fraction": float(np.mean([row["recovery_observed_fraction"] for row in rows])) if rows else float("nan"),
        }

    training_path = output_dir / "training_metrics.csv"
    with training_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "training_seed", "episode_index", "episode_reward", "episode_steps", "loss"],
        )
        writer.writeheader()
        writer.writerows(all_training_rows)

    episodes_path = output_dir / "evaluation_episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "training_seed",
                "evaluation_seed",
                "evaluation_condition",
                "episode_index",
                "success",
                "episode_steps",
                "path_length",
                "wind_change_observed",
                "wind_change_step",
                "episode_ended_before_wind_change",
                "first_odor_reacquisition_step",
                "first_odor_reacquisition_time",
                "tracking_recovery_step",
                "tracking_recovery_time",
                "sustained_odor_detection_proxy_step",
                "sustained_odor_detection_proxy_time",
                "recovery_metric_available",
                "recovery_observed",
                "recovery_not_observed",
                "pre_change_odor_detection_rate",
                "post_change_odor_detection_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(evaluation_records)

    summary_rows = summarize_episode_records(evaluation_records)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    (output_dir / "summary_results.csv").write_bytes(summary_path.read_bytes())

    seed_summary_rows = []
    for method_name in sorted(method_training_rows):
        method_records = [record for record in evaluation_records if record["method"] == method_name]
        by_seed = {}
        for record in method_records:
            by_seed.setdefault(int(record["training_seed"]), []).append(record)
        for training_seed, rows in sorted(by_seed.items()):
            success_rate = float(np.mean([float(row["success"]) for row in rows])) if rows else float("nan")
            mean_steps = float(np.mean([float(row["episode_steps"]) for row in rows])) if rows else float("nan")
            mean_distance = float(np.mean([float(row["path_length"]) for row in rows])) if rows else float("nan")
            recovery_times = [float(row["tracking_recovery_time"]) for row in rows if row["tracking_recovery_time"] is not None]
            seed_summary_rows.append(
                {
                    "method": method_name,
                    "training_seed": training_seed,
                    "evaluation_seed_count": len({int(row["evaluation_seed"]) for row in rows}),
                    "success_rate": success_rate,
                    "mean_steps": mean_steps,
                    "mean_distance": mean_distance,
                    "mean_recovery_steps": float(np.mean(recovery_times)) if recovery_times else float("nan"),
                    "recovery_observed_fraction": float(sum(int(row["recovery_observed"]) for row in rows) / len(rows)) if rows else float("nan"),
                    "wind_change_event_fraction": float(sum(int(row["wind_change_observed"]) for row in rows) / len(rows)) if rows else float("nan"),
                    "episodes_ended_before_wind_change": sum(int(row["episode_ended_before_wind_change"]) for row in rows),
                }
            )

    results_path = output_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "training_seed",
                "evaluation_seed_count",
                "success_rate",
                "mean_steps",
                "mean_distance",
                "mean_recovery_steps",
                "recovery_observed_fraction",
                "wind_change_event_fraction",
                "episodes_ended_before_wind_change",
            ],
        )
        writer.writeheader()
        writer.writerows(seed_summary_rows)

    (output_dir / "episode_results.csv").write_bytes(episodes_path.read_bytes())

    save_plots(config, metrics, samples, all_training_rows, output_dir)
    print(f"\nFinished. Results saved in: {output_dir}")
    print(f"Open {output_dir / 'summary.csv'} and {output_dir / 'evaluation_episodes.csv'} for the run-level evidence.")
    return output_dir
