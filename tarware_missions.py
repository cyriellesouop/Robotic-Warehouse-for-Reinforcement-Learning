"""Run one TA-RWARE mission with FIXED execution rules and record what the trigger may observe.

Nothing here learns. Each AGV gets a queue of orders; each order is five dependent tasks:
  travel to the shelf -> load (needs a picker) -> transport to a station -> return to the shelf's
  storage cell -> unload (needs a picker).
TA-RWARE moves the robots (A*, collisions); this file only tells them where to go:
  - an AGV goes to the target cell of its current task;
  - each picker serves its own section of racks and goes to the AGV that, in its section, has been
    heading to a shelf or waiting for handling the longest.
Task data (nominal duration p, fixed deadline d) is fixed when the mission is built, from the
reference plan; it is never changed during execution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tarware  # noqa: F401  (registers the "tarware-..." environments)
from tarware.utils.utils import flatten_list, split_list

ENV_ID = "tarware-tiny-3agvs-2pickers-globalobs-v1"
ORDERS_PER_AGV = 3
TURN_FACTOR = 1.3     # nominal travel = A* path length x 1.3 (turns cost steps; measured ~1.3)
HANDLING_STEPS = 10   # nominal duration of a load / unload, in steps (measured median wait)
SLACK = 0.3           # deadline generator only: d = reference finish x (1 + SLACK)


@dataclass
class Task:
    agv: int                    # index of the AGV in the agents list
    kind: str                   # "travel" or "handling"
    cell: Tuple[int, int]       # target cell (x, y)
    loc_id: int                 # TA-RWARE location id of that cell (the AGV's destination action)
    p: float                    # nominal duration (steps)
    deadline: float             # fixed absolute deadline (steps from mission start)
    start: Optional[int] = None
    end: Optional[int] = None   # None: not completed when the mission stopped
    # What the trigger may observe while the task executes and its deadline has not passed:
    # (time t, remaining-time estimate r_hat at t).
    trace: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def missed(self) -> bool:
        """Outcome, known only after the mission: not completed by its deadline."""
        return self.end is None or self.end > self.deadline


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_mission(seed: int) -> List[Task]:
    """Builds a mission, executes it with the fixed rules, and returns every task with its trace."""
    env = gym.make(ENV_ID, disable_env_checker=True)
    env.reset(seed=seed)
    w = env.unwrapped
    rng = np.random.default_rng(seed)
    loc_of = {coords: k for k, coords in w.action_id_to_coords_map.items()}  # (y, x) -> location id
    agvs = w.agents[:w.num_agvs]
    pickers = w.agents[w.num_agvs:]
    sections = [flatten_list(s) for s in split_list(w.rack_groups, len(pickers))]  # (y, x) cells

    # --- Mission: ORDERS_PER_AGV orders per AGV, distinct shelves, stations in turn. -------------
    # One extra shelf per AGV marks its parking cell: where it waits once its queue is done. (An
    # AGV left idle where it unloaded traps the picker on that cell: TA-RWARE then never lets the
    # picker move. No order uses a parking cell, so no picker is ever sent there.)
    shelves = list(rng.choice(w.shelfs, size=(ORDERS_PER_AGV + 1) * len(agvs), replace=False))
    parking = [loc_of[(s.y, s.x)] for s in shelves[ORDERS_PER_AGV * len(agvs):]]
    queues = []
    for a, agv in enumerate(agvs):
        queue, here, ref_finish = [], (agv.x, agv.y), 0.0
        for o in range(ORDERS_PER_AGV):
            shelf = shelves[a * ORDERS_PER_AGV + o]
            station = w.goals[(a * ORDERS_PER_AGV + o) % len(w.goals)]  # goals are (x, y)
            steps = [("travel", (shelf.x, shelf.y)), ("handling", (shelf.x, shelf.y)),
                     ("travel", station), ("travel", (shelf.x, shelf.y)), ("handling", (shelf.x, shelf.y))]
            for kind, cell in steps:
                if kind == "travel":
                    # Nominal travel time: the obstacle-free A* path length from where the AGV is
                    # planned to be (TA-RWARE cells are (x, y); find_path takes (y, x)).
                    path = w.find_path((here[1], here[0]), (cell[1], cell[0]), agv, care_for_agents=False)
                    p = float(max(1, round(TURN_FACTOR * len(path))))
                else:
                    p = float(HANDLING_STEPS)
                ref_finish += p                      # reference plan: tasks back to back
                queue.append(Task(a, kind, cell, loc_of[(cell[1], cell[0])], p, round(ref_finish * (1 + SLACK))))
                here = cell
        queues.append(queue)
    tasks = [t for q in queues for t in q]
    current = [0] * len(agvs)                         # index of each AGV's current task
    for q in queues:
        q[0].start = 0

    # --- Execution with the fixed rules. --------------------------------------------------------
    for t in range(1, w.max_steps + 1):
        actions = [0] * w.num_agents
        for a, q in enumerate(queues):
            # The AGV heads to its current task's cell, or to its parking cell once its queue is done.
            actions[a] = q[current[a]].loc_id if current[a] < len(q) else parking[a]
        for k, picker in enumerate(pickers):
            # Requests in this picker's section: AGVs whose current target is a rack cell of the
            # section, i.e. heading to a shelf (load / return) or waiting there for handling.
            # Stations are not rack cells, so a transport to a station never calls a picker.
            requests = [q[current[a]] for a, q in enumerate(queues) if current[a] < len(q)
                        and (q[current[a]].cell[1], q[current[a]].cell[0]) in sections[k]]
            if requests:
                actions[w.num_agvs + k] = min(requests, key=lambda task: task.start).loc_id
        env.step(actions)

        for a, (agv, q) in enumerate(zip(agvs, queues)):
            # Task ends (several may end in one step, e.g. arrival and loading together).
            while current[a] < len(q):
                task = q[current[a]]
                done = ((agv.x, agv.y) == task.cell if task.kind == "travel"
                        else (agv.carrying_shelf is not None) == (current[a] % 5 == 1))
                if not done:
                    break
                task.end = t
                current[a] += 1
                if current[a] < len(q):
                    q[current[a]].start = t
            # Feedback of the executing task: the remaining-time estimate r_hat.
            if current[a] < len(q):
                task = q[current[a]]
                if task.kind == "travel":
                    r_hat = len(agv.path) if agv.path else _distance((agv.x, agv.y), task.cell)
                else:  # waiting for a picker: its distance to the cell, plus the handling step
                    k = next(i for i, s in enumerate(sections) if (task.cell[1], task.cell[0]) in s)
                    r_hat = _distance((pickers[k].x, pickers[k].y), task.cell) + 1
                if t < task.deadline:
                    task.trace.append((t, float(r_hat)))
        if all(current[a] >= len(q) for a, q in enumerate(queues)):
            break
    env.close()
    return tasks
