"""Toy simulation environment: a few robots, each executing ONE task from t = 0.

Deliberately simple, to get the learning loop right before connecting the real DWAFP layers:
  - task j: nominal duration p ~ U[20, 60] s, deadline d = p (1 + U[0.2, 1.0]);
  - with probability `delay_prob` the robot gets stuck once, after U[0.2, 0.7] x p of work, for
    U[0.2, 1.0] x p seconds;
  - feedback every dt seconds: r_hat = remaining work x U[0.95, 1.05], plus, while stuck, the
    robot's own guess of how long it stays stuck (true value x lognormal noise): an informative
    but inaccurate estimate, never the true future;
  - analysis (the model-based layer, here a stand-in): the task is threatened when the reported
    prediction t + r_hat exceeds d; the only repair is to restart the task on a spare robot after
    a dispatch latency, applied only if it is validated to finish by d.
What is NOT modelled yet: dependencies, queues and propagation to other tasks. Each robot's task
is independent; one shared policy serves all of them.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .interface import AnalysisOutcome
from .state import RuntimeSnapshot, TaskView


@dataclass
class _Truth:
    """Ground truth of one task, known only to the simulator."""

    p: float
    deadline: float
    block_at: float          # work done when the robot gets stuck (inf: never)
    block_left: float        # seconds of being stuck still to come
    work_done: float = 0.0
    repaired_finish: Optional[float] = None  # set when a repair restarts it on a spare robot
    finish: Optional[float] = None


class ToyFleetEnv:
    def __init__(self, n_robots: int = 4, dt: float = 1.0, repair_latency: float = 2.0,
                 delay_prob: float = 0.6, seed: int = 0) -> None:
        self.n_robots = n_robots
        self.dt = dt
        self.repair_latency = repair_latency
        self.delay_prob = delay_prob
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.tasks: Dict[str, _Truth] = {}

    # --- episode -------------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> RuntimeSnapshot:
        if seed is not None:
            self.rng = np.random.default_rng(seed)  # same seed = same tasks: paired comparisons
        self.time = 0.0
        self.tasks = {}
        for r in range(self.n_robots):
            p = float(self.rng.uniform(20, 60))
            deadline = p * (1 + float(self.rng.uniform(0.2, 1.0)))
            delayed = self.rng.random() < self.delay_prob
            block_at = float(self.rng.uniform(0.2, 0.7)) * p if delayed else float("inf")
            block_len = float(self.rng.uniform(0.2, 1.0)) * p if delayed else 0.0
            self.tasks[f"T{r}"] = _Truth(p, deadline, block_at, block_len)
        return self.snapshot()

    def done(self) -> bool:
        return all(t.finish is not None for t in self.tasks.values())

    def step(self) -> Tuple[RuntimeSnapshot, Dict[str, bool]]:
        """Advances time by dt. Returns the new snapshot and the tasks that finished during this
        step: task_id -> True if it missed its deadline."""
        self.time += self.dt
        finished = {}
        for task_id, t in self.tasks.items():
            if t.finish is not None:
                continue
            if t.repaired_finish is not None:
                if self.time >= t.repaired_finish:
                    t.finish = t.repaired_finish
            elif t.work_done >= t.block_at and t.block_left > 0:
                t.block_left -= self.dt  # stuck: no progress
            else:
                t.work_done += self.dt
                if t.work_done >= t.p:
                    t.finish = self.time
            if t.finish is not None:
                finished[task_id] = t.finish > t.deadline
        return self.snapshot(), finished

    def snapshot(self) -> RuntimeSnapshot:
        """The read-only view given to the trigger. A repaired task is handed back to the plan
        (new attempt, new reference) and is no longer monitored in this first version."""
        views = []
        for r, (task_id, t) in enumerate(self.tasks.items()):
            if t.finish is not None or t.repaired_finish is not None:
                continue
            views.append(TaskView(task_id, f"robot{r}", t.p, t.deadline, 0.0, self._report(t)))
        return RuntimeSnapshot(self.time, tuple(views))

    def _report(self, t: _Truth) -> float:
        """The robot's remaining-time estimate r_hat (noisy, never the true future)."""
        remaining = (t.p - t.work_done) * float(self.rng.uniform(0.95, 1.05))
        if t.work_done >= t.block_at and t.block_left > 0:
            remaining += t.block_left * float(np.exp(self.rng.normal(0.0, 0.5)))
        return max(0.0, remaining)

    # --- analysis / repair (stands in for the DWAFP layers) -----------------------------------

    def request_analysis(self, snapshot: RuntimeSnapshot, task_id: str) -> AnalysisOutcome:
        view = next(v for v in snapshot.executing if v.task_id == task_id)
        predicted_finish = snapshot.time + view.remaining_estimate
        if predicted_finish <= view.deadline:
            return AnalysisOutcome(threatened=False, repaired=False)
        # Candidate repair: restart on a spare robot after the dispatch latency (work is lost).
        candidate_finish = snapshot.time + self.repair_latency + view.nominal_duration
        if candidate_finish > view.deadline:
            return AnalysisOutcome(threatened=True, repaired=False)  # validation fails: escalate
        self.tasks[task_id].repaired_finish = candidate_finish
        return AnalysisOutcome(threatened=True, repaired=True)
