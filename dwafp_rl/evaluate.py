"""Evaluation loop: every policy on the SAME missions (same seeds), no learning, no exploration."""

from typing import Dict

from .env import ToyFleetEnv
from .train import EpisodeStats, RewardConfig, run_episode
from .trigger import GuardConfig, TriggerWrapper


def evaluate(policy, episodes: int = 500, n_robots: int = 4, seed: int = 100_000,
             guard: GuardConfig = None, rc: RewardConfig = None) -> Dict[str, float]:
    """Mean results per task. Evaluation seeds start far from the training seeds."""
    rc = rc if rc is not None else RewardConfig()
    guard = guard if guard is not None else GuardConfig(enabled=False)
    if hasattr(policy, "explore"):
        policy.explore = False
    env = ToyFleetEnv(n_robots=n_robots)
    wrapper = TriggerWrapper(policy, env, guard)
    total = EpisodeStats()
    for i in range(episodes):
        s = run_episode(env, wrapper, rc, seed=seed + i)
        for name in ("tasks", "misses", "analyses", "unneeded", "repairs", "guard_activations"):
            setattr(total, name, getattr(total, name) + getattr(s, name))
        total.reward += s.reward
    n = max(1, total.tasks)
    return {
        "miss rate": total.misses / n,
        "analyses/task": total.analyses / n,
        "unneeded/task": total.unneeded / n,
        "repairs/task": total.repairs / n,
        "guard/task": total.guard_activations / n,
        "reward/task": total.reward / n,
    }


def print_table(results: Dict[str, Dict[str, float]]) -> None:
    columns = ["miss rate", "analyses/task", "unneeded/task", "repairs/task", "guard/task", "reward/task"]
    print(f"{'policy':<28}" + "".join(f"{c:>15}" for c in columns))
    for name, row in results.items():
        print(f"{name:<28}" + "".join(f"{row[c]:>15.3f}" for c in columns))
