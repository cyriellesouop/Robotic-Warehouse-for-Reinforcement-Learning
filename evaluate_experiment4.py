
#!/usr/bin/env python3
"""Experiment 4: Evaluate PPO reward sensitivity on identical held-out missions."""

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from tarware_missions import run_mission
from trigger_env import EVAL_MISSIONS, evaluate_tasks, margin_rule, never_alert


MODEL_DIR = Path("training/saved_models")
OUTPUT_DIR = Path("evaluation/experiment4")

CONFIGS = [
    ("R0", 1, 1),
    ("R1", 2, 1),
    ("R2", 1, 2),
    ("R3", 2, 2),
]


def load_tasks(start: int, n_missions: int):
    """Generate one common task dataset for all evaluated policies."""
    missions = EVAL_MISSIONS[start:start + n_missions]

    if len(missions) != n_missions:
        raise ValueError("Requested mission range exceeds EVAL_MISSIONS")

    return [task for mission in missions for task in run_mission(mission)]


def make_policy(model):
    """Return the original deterministic PPO policy."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action.item()) if hasattr(action, "item") else int(action)

    return policy


def evaluate_models(tasks, lead_min: float, seed: int):
    """Evaluate all reward configurations on the same task traces."""
    results = []

    for name, fp, fn in CONFIGS:
        model_path = MODEL_DIR / f"trigger_ppo_lead{lead_min:g}_fp{fp}_fn{fn}_seed{seed}.zip"

        if not model_path.is_file():
            raise FileNotFoundError(f"Missing trained model: {model_path}")

        model = PPO.load(model_path)
        metrics = evaluate_tasks(make_policy(model), tasks, eval_lead_min=lead_min)

        results.append({
            "model": name, "fp_penalty": fp, "fn_penalty": fn,
            "decision_threshold": 0.5, **metrics
        })

    # Original deterministic baseline from previous experiments
    for name, policy in [("Margin rule", margin_rule), ("Always WAIT", never_alert)]:
        metrics = evaluate_tasks(policy, tasks, eval_lead_min=lead_min)
        results.append({
            "model": name, "fp_penalty": "", "fn_penalty": "",
            "decision_threshold": "", **metrics
        })

    return results


def save_results(results, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_results(results):
    print("\n" + "=" * 92)
    print(f"{'Model':<15} {'Precision':>11} {'Recall':>10} {'Timely':>10} {'FAR':>10} {'Lead med.':>12}")
    print("=" * 92)

    for r in results:
        print(
            f"{r['model']:<15} "
            f"{r['precision']:>11.2%} "
            f"{r['recall']:>10.2%} "
            f"{r['timely recall']:>10.2%} "
            f"{r['false-alarm rate']:>10.2%} "
            f"{r['lead time median']:>12.2f}"
        )

    print("=" * 92)

    sample = results[0]
    print(f"\nMonitorable tasks: {sample['monitored tasks']}")
    print(f"Monitorable deadline misses: {sample['monitored misses']}")
    print(f"Excluded (no report): {sample['excluded: no report before deadline']}")
    print(f"Excluded (unknown outcome): {sample['excluded: outcome unknown']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 4 reward configurations.")
    parser.add_argument("--lead-min", type=float, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--n-missions", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.lead_min <= 0 or args.n_missions <= 0:
        parser.error("lead-min and n-missions must be positive")

    start = 150 if args.split == "validation" else 200

    print(f"Experiment 4 — {args.split.upper()}")
    print(f"Mission IDs: {EVAL_MISSIONS[start]} to {EVAL_MISSIONS[start + args.n_missions - 1]}")
    print(f"Evaluation lead time: {args.lead_min}")

    tasks = load_tasks(start, args.n_missions)
    results = evaluate_models(tasks, args.lead_min, args.seed)

    output = args.output_dir / f"reward_sensitivity_{args.split}.csv"
    save_results(results, output)
    print_results(results)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
