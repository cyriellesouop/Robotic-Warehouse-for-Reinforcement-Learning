"""Trigger decision wrapper: policy + independent guard + logging.

For every executing task in a snapshot it builds the observation, asks the policy (learned or
baseline) for WAIT / ANALYZE, lets the guard overrule a WAIT, calls the analysis interface on
ANALYZE, and logs the decision with its SOURCE ("policy" or "guard"), so guard activations are
never mixed up with learned decisions.

Guard (configurable, can be switched off): force ANALYZE when the robot's own estimate says the
task will miss its deadline by more than `margin` x its nominal duration, unless the task was
analyzed less than `cooldown` seconds ago. Assumptions it adds: the remaining-time estimate is
roughly unbiased, and a predicted lateness that large is worth an analysis. It is a fallback
against a policy that waits too long, not a safety guarantee.
"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .agent import ACTION_NAMES, ANALYZE, WAIT
from .interface import AnalysisOutcome, AnalysisRepairInterface
from .observation import ObservationBuilder
from .state import RuntimeSnapshot


@dataclass
class GuardConfig:
    enabled: bool = True
    margin: float = -0.25   # predicted lateness beyond 25 % of the nominal duration
    cooldown: float = 5.0   # seconds between two analyses of the same task forced by the guard


@dataclass(frozen=True)
class Decision:
    """One logged decision for one task at one time."""

    time: float
    task_id: str
    obs: Tuple[float, float, float]   # margin, progress, trend
    action: int
    source: str                       # "policy" or "guard"
    outcome: Optional[AnalysisOutcome]


class TriggerWrapper:
    def __init__(self, policy, analysis: AnalysisRepairInterface, guard: GuardConfig = None) -> None:
        self.policy = policy
        self.analysis = analysis
        self.guard = guard if guard is not None else GuardConfig()
        self.builder = ObservationBuilder()
        self.log: List[Decision] = []
        self._last_analysis: Dict[str, float] = {}  # task_id -> time of its last analysis

    def reset(self) -> None:
        """Starts a new mission: forgets observation histories, analysis times, and any per-task
        state of the policy (task ids repeat from one mission to the next). The log is kept."""
        self.builder = ObservationBuilder()
        self._last_analysis = {}
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def step(self, snapshot: RuntimeSnapshot) -> List[Decision]:
        decisions = []
        for task in snapshot.executing:
            obs = self.builder.build(snapshot.time, task)
            action, source = self.policy.decide(obs, task), "policy"
            if action == WAIT and self._guard_fires(snapshot.time, task.task_id, obs[0]):
                action, source = ANALYZE, "guard"
            outcome = None
            if action == ANALYZE:
                # The only way the trigger acts on the world: a request, answered by the
                # model-based layers. Queues and assignments stay out of its reach.
                outcome = self.analysis.request_analysis(snapshot, task.task_id)
                self._last_analysis[task.task_id] = snapshot.time
                if outcome.repaired:
                    self.builder.forget(task.task_id)
            decision = Decision(snapshot.time, task.task_id, tuple(float(x) for x in obs), action, source, outcome)
            self.log.append(decision)
            decisions.append(decision)
        return decisions

    def _guard_fires(self, now: float, task_id: str, margin: float) -> bool:
        if not self.guard.enabled or margin >= self.guard.margin:
            return False
        last = self._last_analysis.get(task_id)
        return last is None or now - last >= self.guard.cooldown


def write_log_csv(log: List[Decision], path: str) -> None:
    """Writes every decision to a CSV file (one row per task and time step)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "task", "margin", "progress", "trend", "action", "source", "threatened", "repaired"])
        for d in log:
            threatened = "" if d.outcome is None else int(d.outcome.threatened)
            repaired = "" if d.outcome is None else int(d.outcome.repaired)
            writer.writerow([d.time, d.task_id, *[round(x, 4) for x in d.obs], ACTION_NAMES[d.action], d.source, threatened, repaired])
