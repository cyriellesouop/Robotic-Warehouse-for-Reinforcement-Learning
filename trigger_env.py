"""TriggerEnv: standalone failure anticipation on TA-RWARE tasks (WAIT / ALERT, no repair).

Terminology: an EPISODE is ONE MONITORED TASK, not a whole warehouse mission. A mission (3 AGVs,
45 tasks) is executed first by fixed rules (tarware_missions.py); its monitorable tasks are then
served one per episode. This is valid because the trigger never changes what the robots do, so
each task's monitoring is an independent stopping problem.

At each report of the task the policy chooses
  WAIT  (0): keep observing;
  ALERT (1): warn that this task risks missing its deadline (the episode ends).
A task is monitorable if it reports at least once before its deadline AND its outcome is known.
The outcome is used ONLY in the reward and the evaluation, never in the observation.

Observation (3 numbers, from the task's plan data and its remaining-time feedback only):
  margin   = (d - (t + r_hat)) / p   predicted slack to the fixed deadline d, in units of p
  progress = (t - start) / p         how much of the nominal duration has elapsed
  trend    = change of the predicted finish t + r_hat per step since the previous report
Reward (once, at the end of the episode):
  ALERT on a task that misses its deadline : +min(1, lead / LEAD_MIN), lead = d - t_alert
  ALERT on a task that meets its deadline  : -1  (false alarm)
  no ALERT, the task misses its deadline   : -1  (missed warning)
  no ALERT, the task meets its deadline    :  0

Mission sets are reserved and never overlap: TRAIN_MISSIONS for training (split between parallel
copies by split_missions), EVAL_MISSIONS for evaluation.
"""


#Your agent currently learns when to alert about a task that will miss its deadline, rather than learning how to prevent that deadline miss.

#Does your reward function encourage the agent to alert as early as possible while avoiding false alerts, or are there situations where waiting is preferable? 
# That question can be answered by analyzing the reward structure and the gamma=0.99 discount factor together.
import gymnasium as gym
import numpy as np

from tarware_missions import run_mission

WAIT, ALERT = 0, 1
# lead_min  defines how early an alert should be raised before a task's deadline to receive the maximum reward.
LEAD_MIN = 5   # steps of warning before the deadline counted as "early enough" (timely)
TRAIN_MISSIONS = range(0, 1_000_000)
EVAL_MISSIONS = range(1_000_000, 1_100_000)


def monitorable(task) -> bool:
    """The trigger can act on it (a report before the deadline) and it can be judged (outcome known)."""
    return bool(task.trace) and task.missed is not None

# def split_missions(missions: range, n: int):
#     slices = []

#     for i in range(n):
#         start = missions.start + i * size
#         end = missions.start + (i + 1) * size

#         slices.append(range(start, end))

#     return slices

def split_missions(missions: range, n: int):
    if n < 1 or n > len(missions):
        raise ValueError("Invalid number of environments")

    size, remainder = divmod(len(missions), n)

    slices = []
    offset = 0

    for i in range(n):
        count = size + (1 if i < remainder else 0)
        slices.append(missions[offset:offset + count])
        offset += count

    return slices


# each step provides 3 features: margin(Predicted deadline slack), progress(Elapsed time since task start), trend(Rate of change in predicted completion time)
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

    # def __init__(self, missions: range = TRAIN_MISSIONS):
    #     self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    #     self.action_space = gym.spaces.Discrete(2)
    #     self.missions = missions   # the only missions this environment may use (bounded)
    #     self.next_index = 0        # position in `missions`; wraps around at the end
    #     self.pending = []          # monitorable tasks of the last mission, not yet served
     
    def __init__(self, missions: range = TRAIN_MISSIONS, lead_min: float = LEAD_MIN):
        super().__init__()
        if lead_min <= 0:
            raise ValueError("lead_min must be positive")

        self.lead_min = float(lead_min)
        self.observation_space = gym.spaces.Box( -np.inf,np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.missions = missions
        self.next_index = 0
        self.pending = [] 
     
     
     #One episode corresponds to monitoring one task, not an entire mission.
     #An episode can last one step, two steps, or more, depending on the number of reports and when the agent raises an alert.
        #      The episode ends when either:
        # 1. PPO chooses ALERT, regardless of whether the alert is correct.
        # 2. PPO continues monitoring until all reports have been processed.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:      # Gymnasium convention: a given seed restarts deterministically,
            self.next_index, self.pending = seed % len(self.missions), []   # inside `missions`
        while not self.pending:
            mission = self.missions[self.next_index]
            self.next_index = (self.next_index + 1) % len(self.missions)
            self.pending = [t for t in run_mission(mission) if monitorable(t)]
        self.task, self.k = self.pending.pop(0), 0
        return features(self.task, 0), {}

    def step(self, action):
        task = self.task
        if action == ALERT:    # ① agent raises the alert
            #lead: Time remaining before the deadline.
            lead = task.deadline - task.trace[self.k][0]  #   steps of warning left before the deadline
            #reward = min(1.0, lead / LEAD_MIN) if task.missed else -1.0  #        correct alert: up to +1, less when late       false alarm: -1
            reward = min(1.0, lead / self.lead_min) if task.missed else -1.0
            return features(task, self.k), reward, True, False, {"alert": True, "missed": task.missed, "lead": lead}
        self.k += 1    # agent chose WAIT: move to the next report
        if self.k == len(task.trace):   # task completed, or its deadline is reached: no more reports
            reward = -1.0 if task.missed else 0.0         #        missed warning: -1    on-time task, never alerted: 0 (correct)
            return features(task, self.k - 1), reward, True, False, {"alert": False, "missed": task.missed, "lead": None}
        return features(task, self.k), 0.0, False, False, {} # ③ WAIT and more reports to come: reward 0, continue


class RewardTriggerEnv(TriggerEnv):
    """
    Experiment 4: configurable reward penalties.

    Inherits the original environment behavior:
      - observations
      - action space
      - mission generation
      - task selection
      - termination logic

    Changes only:
      - false-positive penalty
      - false-negative penalty
    """

    def __init__(
        self,
        missions=TRAIN_MISSIONS,
        lead_min=LEAD_MIN,
        fp_penalty=1.0,
        fn_penalty=1.0,
    ):
        super().__init__(
            missions=missions,
            lead_min=lead_min,
        )

        self.fp_penalty = float(fp_penalty)
        self.fn_penalty = float(fn_penalty)

        if self.fp_penalty <= 0 or self.fn_penalty <= 0:
            raise ValueError(
                "Penalty magnitudes must be positive"
            )

    def step(self, action):
        """
        Preserve original transitions, then adjust
        the terminal reward based on the outcome.
        """

        obs, reward, terminated, truncated, info = (
            super().step(action)
        )

        if terminated or truncated:

            missed = info.get("missed")
            alerted = info.get("alert", False)

            if alerted and missed is False:
                reward = -self.fp_penalty

            elif not alerted and missed is True:
                reward = -self.fn_penalty

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )


def evaluate_tasks(policy, tasks,eval_lead_min: float = LEAD_MIN) -> dict:
    """Runs `policy` (observation -> 0/1) on every monitorable task of `tasks`; warning metrics.
    Tasks that cannot be monitored are counted, not hidden."""
    env = TriggerEnv(lead_min=eval_lead_min)
    tp = tp_timely = fp = fn = tn = 0
    leads = []
    #total_reward = 0.0                # same reward as in training, to compare with ep_rew_mean
    total_reward = 0.0  # reward under a shared evaluation LEAD_MIN
    for task in filter(monitorable, tasks):
        env.pending = [task]          # serve exactly this task
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, done, _, info = env.step(policy(obs))
            total_reward += reward
        if info["alert"] and info["missed"]:
            tp += 1
            tp_timely += info["lead"] >= eval_lead_min
            leads.append(info["lead"])
        elif info["alert"]:
            fp += 1
        elif info["missed"]:
            fn += 1
        else:
            tn += 1
    misses = tp + fn
    no_trace = [t for t in tasks if not t.trace]
    return {
        "precision": tp / max(1, tp + fp),            # share of alerts that were right
        "recall": tp / max(1, misses),                # share of misses warned (at any time)
        "timely recall": tp_timely / max(1, misses),  # share of misses warned >= LEAD_MIN steps ahead
        "missed-warning rate": fn / max(1, misses),   # share of misses with no warning
        "false-alarm rate": fp / max(1, fp + tn),     # share of on-time tasks wrongly flagged
        "lead time median": float(np.median(leads)) if leads else 0.0,
        "lead time p10": float(np.percentile(leads, 10)) if leads else 0.0,
        "mean reward per task": total_reward / max(1, tp + fp + fn + tn),
        "monitored tasks": tp + fp + fn + tn,
        "monitored misses": misses,
        "excluded: no report before deadline": len(no_trace),
        "  of which missed": sum(t.missed is True for t in no_trace),
        "excluded: outcome unknown": sum(bool(t.trace) and t.missed is None for t in tasks),
    }


def evaluate(policy, n_missions: int = 50, missions: range = EVAL_MISSIONS,eval_lead_min: float = LEAD_MIN) -> dict:
    """evaluate_tasks on the tasks of the first n_missions evaluation missions (same for every policy)."""
    tasks = [t for m in missions[:n_missions] for t in run_mission(m)]
    return evaluate_tasks(policy, tasks,eval_lead_min)


# Baselines with the same observation -> action interface.
def never_alert(obs):
    return WAIT


def margin_rule(obs):
    """ALERT as soon as the robot's own estimate predicts a finish after the deadline."""
    return ALERT if obs[0] < 0 else WAIT
