
#!/usr/bin/env python3
"""
Experiment 2B: First-Report Availability and Warning-Horizon Feasibility.

Analyze when TA-RWARE tasks first become monitorable.

No PPO model is loaded. No training is performed.

Outputs:
    first_report_tasks.csv
    first_report_summary.csv
    first_report_distribution.png
    feasible_recall.png
    report_availability.csv

Example:
    python analyze_first_report.py \
        --n-missions 50 \
        --lead-thresholds 2 3 5 8 10 15 20
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tarware_missions import run_mission
from trigger_env import EVAL_MISSIONS


# ---------------------------------------------------------
# 1. Classify each task
# ---------------------------------------------------------

def classify_task(task):
    """
    Classification based on the final deadline outcome
    and availability of pre-deadline reports.
    """

    if task.missed is None:
        return "unknown_outcome"

    if not task.missed:
        return "deadline_met"

    # The remaining cases are deadline misses.

    if task.trace:
        return "missed_monitorable"

    if task.start is None:
        return "missed_not_started"

    if task.start >= task.deadline:
        return "missed_started_after_deadline"

    return "missed_no_report_other"


# ---------------------------------------------------------
# 2. Extract mission/task information
# ---------------------------------------------------------

def collect_records(n_missions):
    records = []

    for mission_id in EVAL_MISSIONS[:n_missions]:
        tasks = run_mission(mission_id)

        for task_idx, task in enumerate(tasks):
            category = classify_task(task)
            first_time = (
                task.trace[0][0]
                if task.trace else None
            )
            last_time = (
                task.trace[-1][0]
                if task.trace else None
            )
            max_lead = (
                task.deadline - first_time
                if first_time is not None
                else None
            )
            # Time remaining at task start
            start_lead = (
                task.deadline - task.start
                if task.start is not None
                else None
            )
            records.append({
                "mission_id": mission_id,
                "task_id": task_idx,
                "agv": task.agv,
                "phase": task.phase,
                "kind": task.kind,
                "start": task.start,
                "end": task.end,
                "deadline": task.deadline,
                "nominal_duration": task.p,
                "missed": task.missed,
                "num_reports": len(task.trace),
                "first_report_time": first_time,
                "last_report_time": last_time,
                "start_lead": start_lead,
                "max_possible_lead": max_lead,
                "category": category,
            })

    return records
# ---------------------------------------------------------
# 3. Save a CSV
# ---------------------------------------------------------
def save_csv(records, path):
    if not records:
        raise ValueError("No records available")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------
# 4. Compute warning feasibility
# ---------------------------------------------------------

def compute_feasibility(records, thresholds):

    missed = [
        r for r in records
        if r["missed"] is True
    ]

    monitorable_misses = [
        r for r in missed
        if r["category"] == "missed_monitorable"
    ]
    leads = np.array(
        [r["max_possible_lead"]
         for r in monitorable_misses],
        dtype=float
    )
    total_misses = len(missed)
    monitored = len(monitorable_misses)
    results = []

    for threshold in thresholds:
        feasible = int(np.sum(leads >= threshold))
        conditional_recall = (
            feasible / monitored
            if monitored else 0.0
        )

        end_to_end_recall = (
            feasible / total_misses
            if total_misses else 0.0
        )
        results.append({
            "lead_threshold": threshold,
            "feasible_tasks": feasible,
            "monitorable_misses": monitored,
            "total_misses": total_misses,
            "conditional_upper_bound":
                conditional_recall,
            "end_to_end_upper_bound":
                end_to_end_recall,
        })

    return results, leads


# ---------------------------------------------------------
# 5. Plot first-report lead-time distribution
# ---------------------------------------------------------

def plot_distribution(leads, output):

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(np.floor(leads.min()) - 0.5,np.ceil(leads.max()) + 1.5,1)
    ax.hist(leads,bins=bins,color="#26877A",edgecolor="white",alpha=0.9)
    median = np.median(leads)
    ax.axvline(median,color="#C46A36",linestyle="--",linewidth=2,label=f"Median = {median:.1f}")
    ax.set_xlabel("Maximum available lead time (simulation steps)")
    ax.set_ylabel("Number of deadline-missing tasks")
    ax.set_title("First-Report Lead-Time Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# 6. Plot achievable warning recall
# ---------------------------------------------------------

def plot_feasibility(results, output):

    thresholds = [ r["lead_threshold"] for r in results]

    conditional = [
        100 * r["conditional_upper_bound"]
        for r in results
    ]

    end_to_end = [
        100 * r["end_to_end_upper_bound"]
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(thresholds,conditional,marker="o",linewidth=2,color="#26877A",label="Monitorable deadline misses")
    ax.plot(thresholds,end_to_end,marker="s",linewidth=2,linestyle="--",color="#C46A36",label="All deadline misses")
    ax.set_xlabel("Required warning lead time (simulation steps)")
    ax.set_ylabel("Maximum achievable actionable recall (%)")
    ax.set_title("Warning Feasibility Under Report Availability")
    ax.set_ylim(0, 105)
    ax.set_xticks(thresholds)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# 7. Save report availability categories
# ---------------------------------------------------------

def save_categories(records, output):

    counts = Counter( r["category"] for r in records)
    rows = [
        {
            "category": category,
            "count": count
        }
        for category, count in sorted(counts.items())
    ]
    save_csv(rows, output)
    return counts

# ---------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------

def print_summary(records, leads, results, counts):

    total = len(records)
    print("\n======================================")
    print(" EXPERIMENT 2B: FIRST-REPORT ANALYSIS")
    print("======================================")
    print(f"\nTotal tasks: {total}")
    print("\nTask categories:")

    for category, count in sorted(counts.items()):
        print(f"  {category:<35} {count:>5}")

    print("\n--- First-report lead time ---")

    if len(leads):

        print(f"Number of monitorable misses: {len(leads)}")
        print(f"Mean lead time:   {np.mean(leads):.2f}")
        print(f"Median lead time: {np.median(leads):.2f}")
        print(f"P10 lead time:    {np.percentile(leads,10):.2f}")
        print(f"P90 lead time:    {np.percentile(leads,90):.2f}")
        print(f"Minimum:          {np.min(leads):.2f}")
        print(f"Maximum:          {np.max(leads):.2f}")

    print("\n--- Warning feasibility ---")

    print(
        f"{'Threshold':>10} "
        f"{'Feasible':>10} "
        f"{'Conditional %':>16} "
        f"{'End-to-end %':>16}"
    )

    for r in results:

        print(
            f"{r['lead_threshold']:>10g} "
            f"{r['feasible_tasks']:>10} "
            f"{100*r['conditional_upper_bound']:>15.2f}% "
            f"{100*r['end_to_end_upper_bound']:>15.2f}%"
        )


# ---------------------------------------------------------
# 9. Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Analyze first-report availability.")
    parser.add_argument("--n-missions",type=int,default=50)
    parser.add_argument("--lead-thresholds", type=float, nargs="+", default=[2, 3, 5, 8, 10, 15, 20])
    parser.add_argument("--output-dir", type=Path,default=Path("evaluation/first_report"))
    args = parser.parse_args()

    if not 1 <= args.n_missions <= len(EVAL_MISSIONS):
        parser.error("Invalid number of missions")

    if not args.lead_thresholds or any(
        x <= 0 for x in args.lead_thresholds
    ):
        parser.error("All lead thresholds must be positive")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # Generate held-out missions.
    records = collect_records(args.n_missions)

    # Save task-level results.
    save_csv(records, output / "first_report_tasks.csv")
    # Classify reporting availability.
    counts = save_categories(records, output / "report_availability.csv")
    # Compute temporal feasibility.
    results, leads = compute_feasibility( records, args.lead_thresholds)
    save_csv(results, output / "first_report_summary.csv")

    # Generate plots.
    if len(leads):
        plot_distribution(leads, output / "first_report_distribution.png")
        plot_feasibility(results, output / "feasible_recall.png")

    print_summary(records,leads,results,counts)
    print(f"\nResults saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
