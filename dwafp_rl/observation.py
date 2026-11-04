"""Observation construction and normalization: three numbers per executing task.

All three come from the task's own plan data and the robot's remaining-time feedback. Nothing
here runs the impact analysis, and each observation costs O(1).

  margin   = (d_j - (t + r_hat)) / p_j   predicted slack to the task's OWN deadline, in units of
                                          its nominal duration (< 0: predicted to be late)
  progress = (t - start) / p_j            how much of the nominal duration has elapsed
  trend    = change of the predicted finish t + r_hat per second since the previous report
             (about 0 when the task progresses normally, about 1 when the robot is stuck)

Dividing by p_j makes short and long tasks comparable, so one policy can serve every task.
"""

from typing import Dict, Tuple

import numpy as np

from .state import TaskView


class ObservationBuilder:
    """Builds observations; keeps one previous report per task to compute the trend."""

    def __init__(self) -> None:
        # task_id -> (time, predicted finish) of the previous report: the only history kept.
        self._previous: Dict[str, Tuple[float, float]] = {}

    def build(self, now: float, task: TaskView) -> np.ndarray:
        p = task.nominal_duration
        if task.remaining_estimate is None:
            # No feedback: assume the nominal plan (finish = start + p), never earlier than now.
            predicted_finish = max(now, task.start_time + p)
        else:
            predicted_finish = now + task.remaining_estimate
        margin = (task.deadline - predicted_finish) / p
        progress = (now - task.start_time) / p
        previous = self._previous.get(task.task_id)
        if previous is None or now <= previous[0]:
            trend = 0.0  # first report of this task: no trend yet
        else:
            trend = (predicted_finish - previous[1]) / (now - previous[0])
        self._previous[task.task_id] = (now, predicted_finish)
        return np.array([margin, progress, trend], dtype=float)

    def forget(self, task_id: str) -> None:
        """Drops the history of a task (it finished, or it was moved by a repair)."""
        self._previous.pop(task_id, None)
