# RL failure-anticipation trigger on TA-RWARE

A small reinforcement-learning environment for one question: **while a warehouse task is
executing, when should we raise an early warning that it will miss its deadline?**

Robots (AGVs) carry shelves in a simulated warehouse ([TA-RWARE](https://github.com/uoe-agents/task-assignment-robotic-warehouse)).
Every task has a fixed deadline. While a task runs, the robot reports how much time it still
needs. At each report, a learned policy chooses:

| Action | Meaning |
|---|---|
| `WAIT` (0) | keep observing |
| `ALERT` (1) | warn now that this task risks missing its deadline |

The policy is judged on whether its warnings are **correct** and **early enough**, compared with
false alarms, late warnings and missed warnings.

This repository contains only the trigger. It is designed to feed a separate workflow-repair
framework, which is not needed here: the trigger never changes what the robots do.

---

## Contents

| File | Purpose |
|---|---|
| `tarware_missions.py` | Runs one TA-RWARE mission with fixed execution rules and records every task: plan data, the robot's remaining-time reports, and the outcome |
| `trigger_env.py` | `TriggerEnv`, the Gymnasium environment (observation, actions, reward); `evaluate()` and two baseline policies |
| `train_trigger.py` | Training with Stable-Baselines3: PPO, A2C or DQN, optionally on parallel environments |
| `tests/test_trigger_env.py` | Tests of the environment's interface, timing logic and reward |
| `launch_tarware.py` | Example: one AGV executes one order; prints the five tasks' durations; `--render` shows it, `--late-picker` creates a handling delay |
| `dwafp_rl/`, `run_demo.py`, `tests/test_basic.py` | An earlier prototype on a toy simulator (tabular Q-learning). Kept for reference; not used by the TA-RWARE environment |

---

## Installation

Tested on Ubuntu 22.04 with Python 3.10, on a CPU (no GPU needed).

### 1. A virtual environment

```bash
python3 -m venv ~/venvs/rl-trigger
source ~/venvs/rl-trigger/bin/activate
pip install --upgrade pip
```

If ROS 2 is sourced in your shell (it sets `PYTHONPATH`), run `unset PYTHONPATH` first in every
terminal you use for this project. Otherwise ROS's Python packages leak into the environment.

### 2. The Python packages

From this repository's folder:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every package to the tested version: PyTorch (CPU build), Gymnasium,
Stable-Baselines3, NumPy, and the versions TA-RWARE needs (`networkx`, `six`, `pyglet`). Its first
line points pip to PyTorch's own package index, where the CPU build (`torch==2.14.1+cpu`, about
200 MB) is published; the default build would add several GB of NVIDIA libraries.

TA-RWARE and its A* package are deliberately not in this file: they need the special install of
step 3.

### 3. TA-RWARE (cloned separately, outside this repository)

```bash
git clone https://github.com/uoe-agents/task-assignment-robotic-warehouse.git
cd task-assignment-robotic-warehouse
git checkout 6109c33                       # the version this environment was built and tested with
pip install --no-build-isolation "pyastar2d @ git+https://github.com/raulsteleac/pyastar2d_TARWARE"
pip install --no-build-isolation -e .
cd -
```

`--no-build-isolation` is required. TA-RWARE's A* package (`pyastar2d`) downloads NumPy during its
build with a deprecated setuptools call, which fails in pip's isolated build. Without isolation it
uses the NumPy already installed.

TA-RWARE pins `pyglet==1.5.11`, `networkx==3.2.1` and `six==1.16.0`; pip installs those versions.

### 4. Check the installation

From this repository's folder:

```bash
python -m unittest -v        # all tests should pass
```

---

## Quick start

```bash
python launch_tarware.py --render                 # one order, five tasks, in a window
python launch_tarware.py --render --late-picker   # the same with a late picker: handling delay
python train_trigger.py --algo ppo --timesteps 20000 --n-envs 4   # short training run to check it works
```

---

## The environment

### Missions and fixed execution rules (`tarware_missions.py`)

Layout: `tarware-tiny-3agvs-2pickers-globalobs-v1`, a 15 × 14 grid with 3 AGVs, 2 pickers,
48 storage cells and 6 stations. A mission gives each AGV a queue of **3 orders**. Each order is
**5 dependent tasks**, executed by the same AGV:

| # | Task | Kind | Ends when |
|---|---|---|---|
| 1 | travel to the requested shelf | travel | the AGV is on the shelf's cell |
| 2 | load the shelf | handling | the AGV carries the shelf (needs a picker on the same cell) |
| 3 | transport it to a station | travel | the AGV is on the station's cell |
| 4 | return it to its storage cell | travel | the AGV is on the storage cell |
| 5 | unload it | handling | the AGV no longer carries it (needs a picker) |

TA-RWARE does all the movement (A* paths, collisions). The rules only choose destinations:
- **AGVs** go to the target cell of their current task; a task starts when the previous one in the queue ends.
- **Pickers** each serve one section of racks. A picker goes to the AGV that, in its section, has
  been heading to a shelf or waiting for handling the longest.
- **Finished AGVs** park on an unused storage cell. TA-RWARE never lets a picker leave a cell that
  an idle AGV occupies, so an AGV left where it unloaded would block that picker forever.

Nothing in these rules is learned.

### Nominal durations and deadlines (fixed when the mission is built)

| Quantity | Definition |
|---|---|
| nominal duration `p`, travel | obstacle-free A* path length × `TURN_FACTOR` (1.3: turns cost steps) |
| nominal duration `p`, handling | `HANDLING_STEPS` = 10 steps (typical wait for a picker) |
| deadline `d` | absolute mission time by which the task must be completed |

Deadlines are mission data: generated once from the reference plan (tasks back to back) as
`d = reference finish × (1 + SLACK)`, with `SLACK` = 0.3. They are **never moved**, even when a task
starts late. A task **misses** its deadline if it is not completed by `d`, including tasks that
never complete.

The three constants are at the top of `tarware_missions.py`.

### The robot's feedback

At every simulation step, each executing task reports a remaining-time estimate `r_hat`:
- **travel:** the remaining length of the AGV's path (Manhattan distance if it has no path at that moment);
- **handling:** the assigned picker's Manhattan distance to the cell, plus 1 step.

`r_hat` is an estimate, not the true future: congestion, clashes and busy pickers make tasks
longer than reported.

### Episodes

The trigger never changes what the robots do. So each task can be judged on its own:
1. a mission is executed with the fixed rules;
2. its tasks are then served **one per episode**, report by report.

A task is used only if it reports at least once before its deadline; a task that starts after its
own deadline cannot be warned about. An episode ends at the first `ALERT`, or when the task
completes, or when its deadline is reached.

### Observation: `Box(3)`

Computed only from the plan data and the reports received so far:

| Feature | Formula | Meaning |
|---|---|---|
| margin | `(d − (t + r_hat)) / p` | predicted slack to the deadline, in units of the task's duration (negative = predicted late) |
| progress | `(t − start) / p` | share of the nominal duration already elapsed |
| trend | change of `t + r_hat` per step since the previous report | about 0 when progressing as planned, positive when the predicted finish slips |

The task's eventual outcome is **never** part of the observation.

### Action: `Discrete(2)`

`0 = WAIT`, `1 = ALERT`.

### Reward (once, at the end of the episode)

| Situation | Reward |
|---|---|
| `ALERT`, the task misses its deadline | `+min(1, lead / LEAD_MIN)`, with `lead = d − t_alert`: +1 if at least `LEAD_MIN` = 5 steps early, less if later |
| `ALERT`, the task meets its deadline | −1 (false alarm) |
| no `ALERT`, the task misses its deadline | −1 (missed warning) |
| no `ALERT`, the task meets its deadline | 0 |

The outcome is used here, in the reward, and in the evaluation only.

`step()` also returns, at the end of an episode, `info = {"alert", "missed", "lead"}`.

### Seeds and task streams

- **Missions are numbered.** `TriggerEnv(seed=s)` serves the tasks of missions `s, s+1, s+2, …` in order.
- **Resetting with a seed restarts there.** `reset(seed=s)` restarts the stream at mission `s` (Gymnasium convention), so the same seed always gives the same tasks.
- **Training and evaluation never share missions.** Training copy `i` uses missions from `i × 100 000`; evaluation uses missions from `1 000 000`.

### Typical statistics (missions 0–39)

| Measure | Value |
|---|---|
| mission length | about 240 steps; no mission stalls |
| travel, actual / nominal | median 1.00, p90 1.31, p99 2.00 |
| handling, actual / nominal | median 1.20, p90 3.80, p99 5.56 (waiting for busy pickers) |
| tasks monitorable before their deadline | 1546 of 1800 |
| monitorable tasks that miss their deadline | 17 % |

With 17 % of tasks missing, a policy that never alerts scores about −0.17 per task. That's the level to beat.

---

## Training (`train_trigger.py`)

### Which algorithms apply

| Algorithm (Stable-Baselines3) | Action spaces accepted | Here |
|---|---|---|
| PPO, A2C (on-policy) | Box, Discrete, MultiDiscrete, MultiBinary | usable |
| DQN (off-policy, replay buffer) | Discrete | usable |
| SAC, TD3, DDPG | Box (continuous) only | not usable |

The script checks the spaces before training and stops on a mismatch.

### Commands

```bash
python train_trigger.py --algo ppo --timesteps 200000 --n-envs 4 --seed 0
python train_trigger.py --algo a2c --timesteps 200000 --n-envs 4 --seed 0
python train_trigger.py --algo dqn --timesteps 200000 --n-envs 4 --seed 0
```

| Option | Meaning | Default |
|---|---|---|
| `--algo` | `ppo`, `a2c` or `dqn` | `ppo` |
| `--timesteps` | total environment steps (one step = one WAIT / ALERT decision) | 200000 |
| `--n-envs` | parallel environments, each in its own process with its own missions | 1 |
| `--seed` | seed of the network initialisation and the exploration | 0 |

**Output:** `models/trigger_<algo>_seed<seed>.zip`.

**During training:** `rollout/ep_rew_mean` is the average reward per task. It should rise above
about −0.17 (the never-alert level).

**About seeds:** the seed is applied globally, not passed to the algorithm. Stable-Baselines3 would
otherwise reset every parallel copy with `seed + i`, which would make the copies replay nearly the
same missions.

**Parallel environments:** each copy is a separate process. Use a few (2–8), depending on your CPU
cores and free memory.

---

## Evaluation (current state)

`trigger_env.evaluate(policy, n_tasks=2000)` runs a policy (observation → 0/1) on 2000 tasks from
fresh missions, and returns:

| Metric | Meaning |
|---|---|
| precision | share of alerts that were right |
| recall | share of missed deadlines that were warned |
| missed-warning rate | share of missed deadlines with no warning |
| false-alarm rate | share of on-time tasks wrongly flagged |
| lead time (median, p10) | steps between a correct alert and the deadline |

Baselines in the same file: `never_alert` and `margin_rule` (alert as soon as the reported estimate
predicts a finish after the deadline).

Evaluating a trained model:

```python
from stable_baselines3 import PPO
from trigger_env import evaluate, margin_rule, never_alert

model = PPO.load("models/trigger_ppo_seed0")
print(evaluate(lambda obs: int(model.predict(obs, deterministic=True)[0])))
print(evaluate(margin_rule))
```

A dedicated evaluation script will follow.

---

## Design rules (please keep them when changing the code)

1. **The trigger only observes.** It never changes assignments, queues or robot behaviour.
2. **No future information in observations.** The outcome may shape the reward and the metrics only.
3. **Deadlines are fixed mission data.** Never recompute them from actual start times.
4. **TA-RWARE stays unmodified.** All behaviour is in this repository's rules, so TA-RWARE can be
   updated or replaced.
5. **Training and evaluation use disjoint missions.**

---

## TA-RWARE quirks handled here

- **Non-standard API.** `reset()` returns only the observations, and rewards and flags are per-robot
  lists. Create TA-RWARE environments with `gym.make(..., disable_env_checker=True)`.
- **Global random generator.** TA-RWARE seeds NumPy's global generator in `reset(seed)`. This code
  uses its own generator for mission content.
- **Handling needs a picker.** With pickers present, an AGV can load or unload only when a picker is on the same cell.
- **Trapped pickers.** A picker cannot leave a cell occupied by an idle AGV; hence the parking rule.

---

## Limitations and next steps

- **One layout,** the tiny one: 3 AGVs, 2 pickers.
- **Independent orders.** Delays spread along each AGV's queue and through shared pickers, but there are no dependencies between orders of different AGVs.
- **Warning only.** The trigger raises warnings and nothing repairs the plan. Evaluating warnings followed by repairs is a later stage.
- **Next:** a dedicated evaluation script; training several seeds per algorithm for a fair comparison.
