"""RL agent: tabular Q-learning over a small grid of the three observations.

The trigger is an optimal-stopping problem per task: at every report, WAIT (keep watching) or
ANALYZE (ask the model-based layers). A table is enough for three features and two actions, it
can be inspected cell by cell, and it needs nothing but NumPy.
"""

import numpy as np

from .state import TaskView

WAIT, ANALYZE = 0, 1
ACTION_NAMES = {WAIT: "WAIT", ANALYZE: "ANALYZE"}

# Bin edges of the grid (values between two edges fall in the same cell).
MARGIN_EDGES = np.array([-0.2, 0.0, 0.1, 0.2, 0.4])   # 6 bins: late ... comfortable
PROGRESS_EDGES = np.array([0.25, 0.5, 0.75, 1.0])     # 5 bins: early ... overrunning
TREND_EDGES = np.array([0.5])                         # 2 bins: progressing / stuck
N_STATES = (len(MARGIN_EDGES) + 1) * (len(PROGRESS_EDGES) + 1) * (len(TREND_EDGES) + 1)


def discretize(obs: np.ndarray) -> int:
    """Maps an observation [margin, progress, trend] to one cell index in [0, N_STATES)."""
    m = int(np.searchsorted(MARGIN_EDGES, obs[0], side="right"))
    p = int(np.searchsorted(PROGRESS_EDGES, obs[1], side="right"))
    t = int(np.searchsorted(TREND_EDGES, obs[2], side="right"))
    return (m * (len(PROGRESS_EDGES) + 1) + p) * (len(TREND_EDGES) + 1) + t


class QTableAgent:
    """Epsilon-greedy tabular Q-learning (no discount: an episode is one task attempt)."""

    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, seed: int = 0) -> None:
        self.q = np.zeros((N_STATES, 2))
        self.alpha = alpha
        self.epsilon = epsilon
        self.explore = True  # False during evaluation and deployment: greedy, no random actions
        self.rng = np.random.default_rng(seed)

    def decide(self, obs: np.ndarray, task: TaskView) -> int:
        if self.explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        s = discretize(obs)
        # Ties (e.g. an unvisited cell, both values 0) resolve to WAIT, the conservative default.
        return ANALYZE if self.q[s, ANALYZE] > self.q[s, WAIT] else WAIT

    def update(self, s: int, a: int, reward: float, s_next, done: bool) -> None:
        """One Q-learning step: Q(s,a) += alpha * (r + max_a' Q(s',a') - Q(s,a))."""
        target = reward if done else reward + float(np.max(self.q[s_next]))
        self.q[s, a] += self.alpha * (target - self.q[s, a])
