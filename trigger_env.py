"""TriggerEnv: standalone failure anticipation on TA-RWARE tasks (WAIT / ALERT, no repair).

An episode monitors ONE executing task, report by report. At each report the policy chooses
  WAIT  (0): keep observing;
  ALERT (1): warn that this task risks missing its deadline (the episode ends).
The trigger never changes what the robots do, so each task can be judged on its own: the missions
are executed first by fixed rules (tarware_missions.py), and their tasks are then served one by
one. The task's eventual outcome is used ONLY in the reward and the evaluation, never in the
observation.

Observation (3 numbers, from the task's plan data and its remaining-time feedback only):
  margin   = (d - (t + r_hat)) / p   predicted slack to the fixed deadline d, in units of p
  progress = (t - start) / p         how much of the nominal duration has elapsed
  trend    = change of the predicted finish t + r_hat per step since the previous report
Reward (once, at the end of the episode):
  ALERT on a task that misses its deadline : +min(1, lead / LEAD_MIN), lead = d - t_alert
                                             (+1 if early enough, less if late)
  ALERT on a task that meets its deadline  : -1  (false alarm)
  no ALERT, the task misses its deadline   : -1  (missed warning)
  no ALERT, the task meets its deadline    :  0
"""

import gymnasium as gym
import numpy as np

from tarware_missions import run_mission

WAIT, ALERT = 0, 1
LEAD_MIN = 5   # steps of warning before the deadline counted as "early enough"


def features(task, k):
    """Observation at the task's k-th report: uses reports 0..k only (no future information)."""
    t, r_hat = task.trace[k]
    margin = (task.deadline - (t + r_hat)) / task.p
    progress = (t - task.start) / task.p
    trend = 0.0
    if k > 0:
        t0, r0 = task.trace[k - 1]
        trend = ((t + r_hat) - (t0 + r0)) / (t - t0)
    return np.array([margin, progress, trend], dtype=np.float32)


class TriggerEnv(gym.Env):
    def __init__(self, seed: int = 0):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.next_mission = seed   # missions are generated in seed order: seed, seed + 1, ...
        self.pending = []          # monitorable tasks of the last mission, not yet served

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:      # Gymnasium convention: a given seed restarts deterministically,
            self.next_mission, self.pending = seed, []   # here from mission number `seed`
        while not self.pending:   # a task is monitorable if it reports before its deadline
            self.pending = [t for t in run_mission(self.next_mission) if t.trace]
            self.next_mission += 1
        self.task, self.k = self.pending.pop(0), 0
        return features(self.task, 0), {}

    def step(self, action):
        task = self.task
        if action == ALERT:
            lead = task.deadline - task.trace[self.k][0]
            reward = min(1.0, lead / LEAD_MIN) if task.missed else -1.0
            return features(task, self.k), reward, True, False, {"alert": True, "missed": task.missed, "lead": lead}
        self.k += 1
        if self.k == len(task.trace):   # task completed, or its deadline is reached: no more reports
            reward = -1.0 if task.missed else 0.0
            return features(task, self.k - 1), reward, True, False, {"alert": False, "missed": task.missed, "lead": None}
        return features(task, self.k), 0.0, False, False, {}


def evaluate(policy, n_tasks: int = 2000, seed: int = 1_000_000) -> dict:
    """Runs `policy` (observation -> 0/1) on n_tasks tasks of fresh missions; warning metrics."""
    env = TriggerEnv(seed)
    tp = fp = fn = tn = 0
    leads = []
    for _ in range(n_tasks):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
        if info["alert"] and info["missed"]:
            tp += 1
            leads.append(info["lead"])
        elif info["alert"]:
            fp += 1
        elif info["missed"]:
            fn += 1
        else:
            tn += 1
    return {
        "precision": tp / max(1, tp + fp),            # share of alerts that were right
        "recall": tp / max(1, tp + fn),               # share of misses that were warned
        "missed-warning rate": fn / max(1, tp + fn),  # share of misses with no warning
        "false-alarm rate": fp / max(1, fp + tn),     # share of on-time tasks wrongly flagged
        "lead time median": float(np.median(leads)) if leads else 0.0,
        "lead time p10": float(np.percentile(leads, 10)) if leads else 0.0,
        "misses / tasks": (tp + fn) / n_tasks,
    }


# Baselines with the same observation -> action interface.
def never_alert(obs):
    return WAIT


def margin_rule(obs):
    """ALERT as soon as the robot's own estimate predicts a finish after the deadline."""
    return ALERT if obs[0] < 0 else WAIT
