"""Baseline trigger policies, with the same decide(obs, task) interface as the agent."""

from typing import Dict

import numpy as np

from .agent import ANALYZE, WAIT
from .state import TaskView


class NeverAnalyze:
    """Reactive: never asks for analysis (the task fails or succeeds on its own)."""

    def decide(self, obs: np.ndarray, task: TaskView) -> int:
        return WAIT


class MarginRule:
    """ANALYZE as soon as the predicted margin to the own deadline drops below a threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def decide(self, obs: np.ndarray, task: TaskView) -> int:
        return ANALYZE if obs[0] < self.threshold else WAIT


class PersistenceRule:
    """The current DWAFP trigger, simplified: ANALYZE when the predicted finish keeps moving
    later (trend > epsilon) for k consecutive reports."""

    def __init__(self, k: int = 3, epsilon: float = 0.05) -> None:
        self.k = k
        self.epsilon = epsilon
        self._count: Dict[str, int] = {}  # task_id -> consecutive reports with trend > epsilon

    def reset(self) -> None:
        """New mission: task ids start over."""
        self._count = {}

    def decide(self, obs: np.ndarray, task: TaskView) -> int:
        count = self._count.get(task.task_id, 0) + 1 if obs[2] > self.epsilon else 0
        self._count[task.task_id] = count
        return ANALYZE if count >= self.k else WAIT
