
"""Evaluate multiple trained triggers on identical held-out missions.

Example:
    python evaluate_trigger.py \
        --models training/saved_models/trigger_ppo_lead*.zip \
        --n-missions 50 \
        --eval-lead-min 5

All policies use the same evaluation tasks and lead-time threshold.
"""

import argparse
import csv
from html import parser
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO

from tarware_missions import run_mission
from trigger_env import (
    EVAL_MISSIONS,
    LEAD_MIN,
    evaluate_tasks,
    margin_rule,
    never_alert,
)

RESULTS_DIR = (Path(__file__).resolve().parent / "evaluation" / "results")

ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

COLUMNS = [
    "mean reward per task",
    "precision",
    "recall",
    "timely recall",
    "missed-warning rate",
    "false-alarm rate",
    "lead time median",
    "lead time p10",
]


def load_policy(model_path: Path):
    """Load a trained model with deterministic action selection."""
    algo = model_path.name.split("_")[1]

    if algo not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model = ALGORITHMS[algo].load(model_path)

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return policy


def evaluation_tasks(n_missions: int):
    """Generate held-out tasks once for all policies."""
    return [
        task
        for m in EVAL_MISSIONS[:n_missions]
        for task in run_mission(m)
    ]


def print_table(results):
    print(f"{'policy':<32}" + "".join(f"{c:>22}" for c in COLUMNS))

    for name, metrics in results.items():
        print(
            f"{name:<32}" +
            "".join(
                f"{metrics[c]:>22.3f}"
                for c in COLUMNS
            )
        )

    m = next(iter(results.values()))

    print(
        f"\nMonitored tasks: {m['monitored tasks']}"
        f" | Missed deadlines: {m['monitored misses']}"
    )

    print("Excluded without report:",m["excluded: no report before deadline"] )

    print("Excluded unknown outcomes:",m["excluded: outcome unknown"])


def save_csv(results, path):
    names = list(next(iter(results.values())).keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["policy"] + [
            name.strip() for name in names
        ])

        for policy, metrics in results.items():
            writer.writerow(
                [policy] + [metrics[name] for name in names]
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate lead-time sensitivity experiments.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", type=Path,nargs="+",help="Paths of trained models")
    group.add_argument("--model",type=Path,help="Single trained model (backward compatibility)" )

    parser.add_argument("--n-missions",type=int,default=50)
    parser.add_argument("--eval-lead-min",type=float,default=LEAD_MIN,help="Common lead-time threshold for evaluation")
    parser.add_argument("--output",type=Path,default=None,help="Optional CSV output path")
    
    args = parser.parse_args()

    model_paths = args.models if args.models else [args.model]

    if args.n_missions < 1 or args.n_missions > len(EVAL_MISSIONS):
        parser.error("--n-missions is outside the evaluation range")

    if args.eval_lead_min <= 0:
        parser.error("--eval-lead-min must be positive")

    for path in model_paths:
        if not path.is_file():
            parser.error(f"Model not found: {path}")

    print("Generating held-out tasks...")
    tasks = evaluation_tasks(args.n_missions)

    print(f"Generated {len(tasks)} tasks")
    print(f"Evaluation LEAD_MIN = {args.eval_lead_min:g}\n")

    # Baselines evaluated on the exact same task list.
    policies = {"always WAIT": never_alert, "margin rule": margin_rule,}

    # Load each trained model.
    for model_path in model_paths:
        name = model_path.stem

        if name in policies:
            parser.error(f"Duplicate policy name: {name}")

        print(f"Loading {name}")
        policies[name] = load_policy(model_path)

    # Apply a common evaluation criterion to all policies.
    results = {}

    for name, policy in policies.items():
        print(f"Evaluating {name}...")

        results[name] = evaluate_tasks(policy,tasks,eval_lead_min=args.eval_lead_min)

    print("\n===== EVALUATION RESULTS =====\n")
    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = args.output

    if output_path is None:
        output_path = (
            RESULTS_DIR /
            f"lead_sensitivity_"
            f"{args.n_missions}missions_"
            f"eval_lead{args.eval_lead_min:g}.csv"
        )

    output_path.parent.mkdir(parents=True,exist_ok=True)

    save_csv(results, output_path)

    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()





















# """Evaluate a trained trigger against the baselines on the same held-out missions.

# Run from this folder in the RL environment:
#     python evaluate_trigger.py --model training/saved_models/trigger_ppo_seed0.zip --n-missions 50
# Three policies, on exactly the same tasks:
#     the trained model, with deterministic actions (no random exploration);
#     always WAIT (never alert);
#     margin rule: ALERT as soon as the estimated finish t + r_hat exceeds the deadline (margin < 0).
# The missions come from EVAL_MISSIONS (1 000 000 on), never used in training. They are generated
# once and every policy is run on the same task list, so differences come from the policies only.
# Output: a table on screen and evaluation/results/<model name>_<n>missions.csv (one row per policy).
# """

# import argparse
# import csv
# from pathlib import Path

# from stable_baselines3 import A2C, DQN, PPO

# from tarware_missions import run_mission
# from trigger_env import EVAL_MISSIONS, evaluate_tasks, margin_rule, never_alert

# RESULTS_DIR = Path(__file__).resolve().parent / "evaluation" / "results"
# ALGORITHMS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}   # chosen from the model's file name
# COLUMNS = ["mean reward per task", "precision", "recall", "timely recall", "missed-warning rate",
#            "false-alarm rate", "lead time median", "lead time p10"]


# def load_policy(model_path: Path):
#     """The trained model as a policy observation -> 0/1, with deterministic actions."""
#     algo = model_path.name.split("_")[1]            # trigger_<algo>_seed<seed>.zip
#     model = ALGORITHMS[algo].load(model_path)
#     return lambda obs: int(model.predict(obs, deterministic=True)[0])


# def evaluation_tasks(n_missions: int):
#     """All tasks of the first n_missions held-out missions (generated once, shared by every policy)."""
#     return [task for m in EVAL_MISSIONS[:n_missions] for task in run_mission(m)]


# def print_table(results: dict) -> None:
#     print(f"{'policy':<30}" + "".join(f"{c:>22}" for c in COLUMNS))
#     for name, m in results.items():
#         print(f"{name:<30}" + "".join(f"{m[c]:>22.3f}" for c in COLUMNS))
#     m = next(iter(results.values()))   # task counts are the same for every policy
#     print(f"\nmonitored tasks: {m['monitored tasks']} (of which missed: {m['monitored misses']}) | "
#           f"excluded, no report before the deadline: {m['excluded: no report before deadline']} "
#           f"(of which missed: {m['  of which missed']}) | excluded, outcome unknown: {m['excluded: outcome unknown']}")


# def save_csv(results: dict, path: Path) -> None:
#     with open(path, "w", newline="") as f:
#         names = list(next(iter(results.values())).keys())
#         writer = csv.writer(f)
#         writer.writerow(["policy"] + [n.strip() for n in names])
#         for policy, m in results.items():
#             writer.writerow([policy] + [m[n] for n in names])


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate a trained trigger against the baselines.")
#     parser.add_argument("--model", type=Path, required=True, help="training/saved_models/trigger_<algo>_seed<seed>.zip")
#     parser.add_argument("--n-missions", type=int, default=50, help="held-out missions (about 39 monitored tasks each)")
#     args = parser.parse_args()

#     tasks = evaluation_tasks(args.n_missions)
#     model_name = args.model.stem                                  # e.g. trigger_ppo_seed0
#     policies = {f"{model_name} (deterministic)": load_policy(args.model),
#                 "always WAIT": never_alert,
#                 "margin rule (finish > deadline)": margin_rule}
#     results = {name: evaluate_tasks(policy, tasks) for name, policy in policies.items()}

#     print(f"held-out missions {EVAL_MISSIONS[0]} to {EVAL_MISSIONS[args.n_missions - 1]}\n")
#     print_table(results)
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#     out = RESULTS_DIR / f"{model_name}_{args.n_missions}missions.csv"
#     save_csv(results, out)
#     print(f"\nsaved {out}")
