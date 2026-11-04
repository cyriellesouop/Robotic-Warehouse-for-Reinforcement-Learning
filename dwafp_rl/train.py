"""Rewards, one episode, and the training loop.

Reward of one task (an episode of the stopping problem), with weights in RewardConfig:
  - each ANALYZE costs `analysis` (computation, attention of the planner);
  - an ANALYZE that finds nothing threatened costs `unneeded` more (an unnecessary request);
  - at the end of the task, `miss` if it finished after its deadline (failure containment).
Timely intervention is not rewarded directly: analysing late simply leaves no feasible repair,
so the miss penalty arrives anyway.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .agent import ANALYZE, QTableAgent, discretize
from .env import ToyFleetEnv
from .trigger import Decision, GuardConfig, TriggerWrapper


@dataclass
class RewardConfig:
    miss: float = 1.0
    analysis: float = 0.02
    unneeded: float = 0.05


@dataclass
class EpisodeStats:
    tasks: int = 0
    misses: int = 0
    analyses: int = 0
    unneeded: int = 0          # analyses that found nothing threatened
    repairs: int = 0
    guard_activations: int = 0  # decisions forced by the guard (logged apart from the policy)
    reward: float = 0.0


def decision_reward(d: Decision, rc: RewardConfig) -> float:
    if d.action != ANALYZE:
        return 0.0
    return -rc.analysis - (rc.unneeded if not d.outcome.threatened else 0.0)


def run_episode(env: ToyFleetEnv, wrapper: TriggerWrapper, rc: RewardConfig, seed: int,
                agent: Optional[QTableAgent] = None) -> EpisodeStats:
    """Runs one mission. With `agent`, the Q-table is updated after every transition."""
    stats = EpisodeStats(tasks=env.n_robots)
    wrapper.reset()
    snapshot = env.reset(seed)
    # task_id -> (state, action, reward) of its last decision, waiting for the next state.
    pending: Dict[str, Tuple[int, int, float]] = {}
    while not env.done():
        for d in wrapper.step(snapshot):
            s = discretize(np.array(d.obs))
            if agent is not None and d.task_id in pending:
                ps, pa, pr = pending[d.task_id]
                agent.update(ps, pa, pr, s, done=False)
            r = decision_reward(d, rc)
            pending[d.task_id] = (s, d.action, r)
            stats.reward += r
            if d.action == ANALYZE:
                stats.analyses += 1
                stats.unneeded += int(not d.outcome.threatened)
                stats.repairs += int(d.outcome.repaired)
                stats.guard_activations += int(d.source == "guard")
        snapshot, finished = env.step()
        for task_id, missed in finished.items():
            ps, pa, pr = pending.pop(task_id)
            final = -rc.miss if missed else 0.0
            if agent is not None:
                agent.update(ps, pa, pr + final, None, done=True)
            stats.misses += int(missed)
            stats.reward += final
    return stats


def train(agent: QTableAgent, episodes: int = 3000, n_robots: int = 4, seed: int = 0,
          rc: RewardConfig = None, guard: GuardConfig = None) -> list:
    """Simulation pretraining. The guard is off by default here, so the agent learns on its own;
    epsilon decays from 0.3 to 0.02."""
    rc = rc if rc is not None else RewardConfig()
    guard = guard if guard is not None else GuardConfig(enabled=False)
    env = ToyFleetEnv(n_robots=n_robots)
    wrapper = TriggerWrapper(agent, env, guard)
    agent.explore = True
    history = []
    for i in range(episodes):
        agent.epsilon = max(0.02, 0.3 * (1 - i / episodes))
        history.append(run_episode(env, wrapper, rc, seed=seed + i, agent=agent))
    agent.explore = False
    return history
