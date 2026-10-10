import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiment import run_experiment

DEFAULT_CONFIG = {
    "grid_size": 15,
    "max_steps": 160,
    "training_episodes": 1200,
    "evaluation_episodes": 150,
    "wind_change_step": 55,
    "history_window": 4,
    "seed": 7,
    "training_seeds": [7, 11],
    "evaluation_seeds": [7, 11, 17, 23, 29],
    "change_wind": True,
    "consecutive_detections": 2,
}

QUICK_CONFIG = {
    "grid_size": 9,
    "max_steps": 40,
    "training_episodes": 30,
    "evaluation_episodes": 12,
    "wind_change_step": 15,
    "history_window": 2,
    "seed": 7,
    "training_seeds": [7],
    "evaluation_seeds": [7, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59],
    "change_wind": True,
    "consecutive_detections": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OdorBot prototype experiment.")
    parser.add_argument("--quick", action="store_true", help="Run a compact smoke-test configuration.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory where the output files are written.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run_experiment(config, output_dir=output_dir)
