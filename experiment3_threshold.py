
#!/usr/bin/env python3
"""
Experiment 3: PPO ALERT Decision-Threshold Sensitivity.

threshold est la valeur à partir de laquelle l'agent décide de déclencher une alerte (ALERT) plutôt que de continuer à observer (WAIT)

No retraining. No changes to observation features or rewards.

Validation missions:
    EVAL_MISSIONS[50:100]

Final test missions:
    EVAL_MISSIONS[100:150]

The original 50 evaluation missions are deliberately excluded
from threshold selection and the final test.

Usage:
    python experiment3_threshold.py \
        --model training/saved_models/trigger_ppo_lead2_seed0.zip \
        --eval-lead-min 5 \
        --max-far 0.02
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from tarware_missions import run_mission
from trigger_env import (
    EVAL_MISSIONS,
    evaluate_tasks,
    never_alert,
    margin_rule,
    WAIT,
    ALERT,
)

# theshold is a limit probability value above which determines when  the agent (PPO) decides to trigger an alert (ALERT) or continue observing (WAIT)
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------
# 1. Extract the probability of ALERT
# ---------------------------------------------------------

def alert_probability(model, obs):
    """
    Probability assigned by PPO actor to action ALERT.

    This is a policy action probability, not necessarily
    a calibrated probability of missing the deadline.
    """

    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    with torch.no_grad():
        distribution = model.policy.get_distribution(
            obs_tensor
        )

        probabilities = distribution.distribution.probs

        return float(
            probabilities[0, ALERT].item()
        )


# ---------------------------------------------------------
# 2. Construct the threshold-based policy
# ---------------------------------------------------------

def make_threshold_policy(model, threshold):

    def policy(obs):

        probability = alert_probability(model, obs)

        return (
            ALERT
            if probability >= threshold
            else WAIT
        )

    return policy


# ---------------------------------------------------------
# 3. Generate a fixed evaluation dataset
# ---------------------------------------------------------

def generate_tasks(start, count):

    missions = EVAL_MISSIONS[start:start + count]

    tasks = []

    for mission_id in missions:
        tasks.extend(run_mission(mission_id))

    return tasks


# ---------------------------------------------------------
# 4. Evaluate multiple thresholds
# ---------------------------------------------------------

def evaluate_thresholds(model, tasks, eval_lead_min):

    results = []

    for threshold in THRESHOLDS:

        policy = make_threshold_policy(
            model,
            threshold
        )

        metrics = evaluate_tasks(
            policy,
            tasks,
            eval_lead_min=eval_lead_min
        )

        results.append({
            "threshold": threshold,
            **metrics
        })

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Timely={metrics['timely recall']:.4f} | "
            f"FAR={metrics['false-alarm rate']:.4f}"
        )

    return results


# ---------------------------------------------------------
# 5. Select threshold on validation missions only
# ---------------------------------------------------------

def select_threshold(results, max_far):
    """
    Maximize timely recall subject to a false-alarm limit.

    Tie-breakers:
      1. Higher precision
      2. Lower false-alarm rate
      3. Threshold closest to 0.5
    """

    eligible = [
        r for r in results
        if r["false-alarm rate"] <= max_far
    ]

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda r: (
            r["timely recall"],
            r["precision"],
            -r["false-alarm rate"],
            -abs(r["threshold"] - 0.5)
        )
    )


# ---------------------------------------------------------
# 6. Save CSV
# ---------------------------------------------------------

def save_csv(records, path):

    if not records:
        raise ValueError("No results to save")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------
# 7. Plot sensitivity curves
# ---------------------------------------------------------

def plot_thresholds(results, output):

    thresholds = [
        r["threshold"] for r in results
    ]

    precision = [
        100 * r["precision"] for r in results
    ]

    recall = [
        100 * r["recall"] for r in results
    ]

    timely = [
        100 * r["timely recall"] for r in results
    ]

    far = [
        100 * r["false-alarm rate"] for r in results
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        thresholds, precision,
        marker="o", label="Precision"
    )

    ax.plot(
        thresholds, recall,
        marker="s", label="Recall"
    )

    ax.plot(
        thresholds, timely,
        marker="^", label="Timely recall"
    )

    ax.plot(
        thresholds, far,
        marker="d", label="False-alarm rate"
    )

    ax.axvline(
        0.5,
        linestyle="--",
        alpha=0.6,
        label="Default threshold"
    )

    ax.set_xlabel("ALERT decision threshold")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(0, 105)
    ax.set_title("PPO Decision-Threshold Sensitivity")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# 8. Plot actionable recall / false-alarm tradeoff
# ---------------------------------------------------------

def plot_tradeoff(results, output):

    far = [
        100 * r["false-alarm rate"]
        for r in results
    ]

    timely = [
        100 * r["timely recall"]
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(far, timely, marker="o")

    for r in results:

        ax.annotate(
            f"{r['threshold']:.1f}",
            (
                100 * r["false-alarm rate"],
                100 * r["timely recall"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlabel("False-alarm rate (%)")
    ax.set_ylabel("Timely recall (%)")

    ax.set_title(
        "Actionable Detection vs False Alarms"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# 9. Final evaluation on untouched test missions
# ---------------------------------------------------------

def final_evaluation(model, threshold, tasks, lead_min):

    policies = {
        "PPO selected threshold":
            make_threshold_policy(model, threshold),

        "PPO default threshold":
            make_threshold_policy(model, 0.5),

        "Margin rule":
            margin_rule,

        "Always WAIT":
            never_alert
    }

    results = []

    for name, policy in policies.items():

        metrics = evaluate_tasks(
            policy,
            tasks,
            eval_lead_min=lead_min
        )

        results.append({
            "policy": name,
            "decision_threshold": (
                threshold
                if name == "PPO selected threshold"
                else 0.5
                if name == "PPO default threshold"
                else ""
            ),
            **metrics
        })

    return results


# ---------------------------------------------------------
# 10. Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--eval-lead-min",
        type=float,
        default=5
    )

    parser.add_argument(
        "--max-far",
        type=float,
        default=0.02,
        help="Maximum acceptable false-alarm rate"
    )

    parser.add_argument(
        "--val-start",
        type=int,
        default=50
    )

    parser.add_argument(
        "--val-missions",
        type=int,
        default=50
    )

    parser.add_argument(
        "--test-start",
        type=int,
        default=100
    )

    parser.add_argument(
        "--test-missions",
        type=int,
        default=50
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/experiment3")
    )

    args = parser.parse_args()

    if not args.model.is_file():
        parser.error("Model file not found")

    if args.eval_lead_min <= 0:
        parser.error("eval-lead-min must be positive")

    if not 0 <= args.max_far <= 1:
        parser.error("max-far must be between 0 and 1")

    if (
        args.val_start < 0
        or args.test_start < 0
        or args.val_missions <= 0
        or args.test_missions <= 0
    ):
        parser.error("Invalid mission ranges")

    val_end = args.val_start + args.val_missions
    test_end = args.test_start + args.test_missions

    if max(val_end, test_end) > len(EVAL_MISSIONS):
        parser.error("Mission range exceeds EVAL_MISSIONS")

    # Do not allow overlapping validation and test IDs.
    if max(args.val_start, args.test_start) < min(
        val_end, test_end
    ):
        parser.error(
            "Validation and test mission ranges overlap"
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    print("Loading PPO model...")
    model = PPO.load(args.model)

    # ---------------- VALIDATION ----------------

    print("\nGenerating validation missions...")

    validation_tasks = generate_tasks(
        args.val_start,
        args.val_missions
    )

    print("\n=== THRESHOLD SWEEP ===")

    validation_results = evaluate_thresholds(
        model,
        validation_tasks,
        args.eval_lead_min
    )

    save_csv(
        validation_results,
        output / "threshold_validation.csv"
    )

    plot_thresholds(
        validation_results,
        output / "threshold_sensitivity.png"
    )

    plot_tradeoff(
        validation_results,
        output / "threshold_tradeoff.png"
    )

    selected = select_threshold(
        validation_results,
        args.max_far
    )

    if selected is None:
        print(
            "\nNo threshold satisfies the "
            "specified false-alarm limit."
        )
        print(
            "Validation results saved. "
            "No final test was performed."
        )
        return

    best_threshold = selected["threshold"]

    print("\n=== SELECTED THRESHOLD ===")

    print("Threshold:", best_threshold)
    print(
        "Validation timely recall:",
        selected["timely recall"]
    )
    print(
        "Validation false-alarm rate:",
        selected["false-alarm rate"]
    )

    # ---------------- FINAL TEST ----------------

    print("\nGenerating final test missions...")

    test_tasks = generate_tasks(
        args.test_start,
        args.test_missions
    )

    test_results = final_evaluation(
        model,
        best_threshold,
        test_tasks,
        args.eval_lead_min
    )

    save_csv(
        test_results,
        output / "threshold_final_test.csv"
    )

    print("\n=== FINAL TEST RESULTS ===")

    for r in test_results:

        print(
            f"{r['policy']:<26} "
            f"Precision={r['precision']:.3f} "
            f"Recall={r['recall']:.3f} "
            f"Timely={r['timely recall']:.3f} "
            f"FAR={r['false-alarm rate']:.3f}"
        )

    print(f"\nResults saved in: {output.resolve()}")


if __name__ == "__main__":
    main()
