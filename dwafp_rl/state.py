"""Mission and runtime-state data structures seen by the trigger.

Both classes are frozen dataclasses holding tuples, so the trigger receives a read-only copy:
it cannot change a deadline, an assignment or a queue, even by accident.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TaskView:
    """What the trigger may know about one executing task (times in seconds)."""

    task_id: str
    robot_id: str
    nominal_duration: float      # p_j, known from the mission (same on every robot)
    deadline: float              # d_j, absolute hard completion deadline
    start_time: float            # start of the current attempt
    remaining_estimate: Optional[float]  # robot feedback r_hat(t); None if not reported


@dataclass(frozen=True)
class RuntimeSnapshot:
    """The executing tasks at one time step (one entry per busy robot)."""

    time: float
    executing: Tuple[TaskView, ...]
