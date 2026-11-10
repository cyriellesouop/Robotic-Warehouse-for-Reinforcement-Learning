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
| `tarware_missions.py` | Runs one TA-RWARE mission with fixed execution rules and records every task: plan data, the robot's remaining-time reports, and the outcome. `python tarware_missions.py` prints a timing check |
| `trigger_env.py` | `TriggerEnv`, the Gymnasium environment (observation, actions, reward); `evaluate()` and two baseline policies |
| `train_trigger.py` | Training with Stable-Baselines3: PPO, A2C or DQN, optionally on parallel environments |
| `evaluate_trigger.py` | Evaluates trained models against `always WAIT` and `margin rule` on the same held-out missions |
| `run_lead_experiments.sh`, `run_lead_evaluation.sh` | `LEAD_MIN` sweep (2–20): training, then evaluation at each warning horizon |
| `analyze_first_report.py` | Earliest possible warning of each missed task (no model needed); outputs in `evaluation/first_report/` |
| `experiment3_threshold.py`, `run_experiment3.sh` | Experiment 3: ALERT-probability threshold of a trained PPO, chosen on validation missions and tested on separate ones; outputs in `evaluation/experiment3/` |
| `evaluation/results/` | Evaluation tables (CSV) |
| `report/` | Experiment reports, figures and the scripts that draw them |
| `training/logs/` | One sub-folder per training run (`<algo>_lead<L>_seed<seed>/`): the training table (`progress.csv`) and every episode's reward and length (`<i>.monitor.csv`) |
| `training/saved_models/` | Every trained model (`trigger_<algo>_lead<L>_seed<seed>.zip`) |
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

| Phase | Task | Kind | Ends when |
|---|---|---|---|
| 0 | travel to the requested shelf | travel | the AGV is on the shelf's cell |
| 1 | load the shelf | handling | the AGV carries the shelf (needs a picker on the same cell) |
| 2 | transport it to a station | travel | the AGV is on the station's cell |
| 3 | return it to its storage cell | travel | the AGV is on the storage cell |
| 4 | unload it | handling | the AGV no longer carries it (needs a picker) |

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
| nominal duration `p`, travel | obstacle-free A* path length × `TURN_FACTOR` (1.3: turns cost steps), from where the AGV is planned to be |
| nominal duration `p`, handling | `HANDLING_STEPS` = 10 steps (typical wait for a picker) |
| deadline `d` | absolute mission time by which the task must be completed |

Deadlines are mission data: generated once from the reference plan (tasks back to back) as
`d = reference finish × (1 + SLACK)`, with `SLACK` = 0.3. They are **never moved**, even when a task
starts late.

**Outcome of a task** (known only after the mission):

| Case | Outcome |
|---|---|
| completed at or before `d` | met |
| completed after `d` | missed |
| not completed, and the mission ran past `d` | missed |
| not completed, and the mission stopped before `d` | **unknown**: excluded from training and evaluation, and counted |

The three constants are at the top of `tarware_missions.py`.

**Loaded or empty routes:** TA-RWARE's A* ignores whether an AGV carries a shelf (only pickers are
restricted to highways). So the routes of loaded trips (transport, return) can be planned while the
AGV is still empty; the timing check confirms it (see the statistics below).

### The robot's feedback

At every simulation step, each executing task reports a remaining-time estimate `r_hat`, with the
**same timing model as the nominal duration**:
- **travel:** `TURN_FACTOR` × the cells left on the AGV's path (Manhattan distance if it has no path at that moment);
- **handling:** the section picker's Manhattan distance to the cell, plus 1 step.

The handling estimate is deliberately simple. It assumes the picker comes straight to this cell,
and **ignores its earlier commitments** (another AGV it serves first). That makes it optimistic;
its error is measured (see the statistics below and `python tarware_missions.py`).

`r_hat` is an estimate, not the true future: congestion, clashes and busy pickers make tasks
longer than reported.

### Episodes

**Terminology: a training episode is one monitored task, not a whole warehouse mission.** One
timestep is one WAIT / ALERT decision, at one report of that task.

This is valid because the trigger never changes what the robots do, so each task is an independent
stopping problem:
1. a mission (3 AGVs, 45 tasks) is executed with the fixed rules;
2. its monitorable tasks are then served one per episode, report by report.

A task is **monitorable** if it reports at least once before its deadline (a task that starts
after its own deadline cannot be warned about) **and** its outcome is known. An episode ends at the
first `ALERT`, when the task completes, or when its deadline is reached.

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
| `ALERT`, the task misses its deadline | `+min(1, lead / LEAD_MIN)`, with `lead = d − t_alert`: +1 if at least `LEAD_MIN` steps early (default 5, `--lead-min`), less if later |
| `ALERT`, the task meets its deadline | −1 (false alarm) |
| no `ALERT`, the task misses its deadline | −1 (missed warning) |
| no `ALERT`, the task meets its deadline | 0 |

The outcome is used here, in the reward, and in the evaluation only.

`step()` also returns, at the end of an episode, `info = {"alert", "missed", "lead"}`.

### Mission sets

Missions are numbered (the number seeds the mission). Two sets are reserved and never overlap:

| Set | Missions | Used by |
|---|---|---|
| `TRAIN_MISSIONS` | 0 – 999 999 | training |
| `EVAL_MISSIONS` | 1 000 000 – 1 099 999 | evaluation |

- **`TriggerEnv(missions)`** uses only the missions of the range it is given, in order. It starts again at the beginning of the range if it ever reaches the end, so it never leaves its range.
- **Parallel training copies** each get one disjoint, bounded slice of `TRAIN_MISSIONS` (`split_missions`). No copy can reach the evaluation missions, whatever their number.
- **`reset(seed=s)`** restarts at position `s` of the environment's range (Gymnasium convention), so the same seed always gives the same tasks.

### Typical statistics (missions 0–39)

| Measure | Value |
|---|---|
| mission length | about 240 steps; no mission stalls; no unknown outcome |
| monitorable tasks | 1546 of 1800 (254 tasks report nothing before their deadline) |
| monitorable tasks that miss their deadline | 17 % |

Timing check (`python tarware_missions.py`). Actual ÷ nominal duration of completed tasks; error
of `r_hat` = `r_hat` − true remaining time, over every report:

| Phase | actual / nominal (median, p90) | `r_hat` bias | `r_hat` mean absolute error |
|---|---|---|---|
| travel to shelf (empty) | 1.00, 1.50 | +0.0 | 1.1 |
| load | 1.45, 4.20 | −10.7 | 10.7 |
| transport to station (loaded) | 0.92, 1.13 | +0.6 | 1.5 |
| return to storage (loaded) | 1.00, 1.30 | +0.3 | 1.6 |
| unload | 0.80, 3.21 | −8.3 | 8.3 |

Travel estimates are accurate, empty or loaded. Handling estimates are about 8–11 steps too
optimistic, because they ignore the picker's earlier commitments.

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
| `--lead-min` | `LEAD_MIN` of the reward | 5 |

**Outputs** (one run = one algorithm, `LEAD_MIN` and seed):

| File | Content |
|---|---|
| `training/saved_models/trigger_<algo>_lead<L>_seed<seed>.zip` | the trained model |
| `training/logs/<algo>_lead<L>_seed<seed>/progress.csv` | the training table printed on screen, one row per print |
| `training/logs/<algo>_lead<L>_seed<seed>/<i>.monitor.csv` | reward (`r`), length (`l`) and time (`t`) of every episode (task) of copy `i` |

Both folders are kept in the repository (empty); their contents are not committed.

**During training:** `rollout/ep_rew_mean` is the average reward per episode (per task) over the
last 100 episodes. It should rise above about −0.17 (the never-alert level).

**About seeds:** the seed is applied globally, not passed to the algorithm. Stable-Baselines3 would
otherwise reset every parallel copy with `seed + i`, moving it to another position of its slice.

**Parallel environments:** each copy is a separate process. Use a few (2–8), depending on your CPU
cores and free memory.

---

## Evaluation

`trigger_env.evaluate(policy, n_missions=50)` runs a policy (observation → 0/1) on every
monitorable task of the first 50 evaluation missions (about 1950 tasks), the same for every policy.
It returns:

| Metric | Meaning |
|---|---|
| precision | share of alerts that were right |
| recall | share of missed deadlines warned, at any time before the deadline |
| **timely recall** | share of missed deadlines warned at least `LEAD_MIN` steps before the deadline (default 5, `eval_lead_min`): useful anticipation |
| missed-warning rate | share of missed deadlines with no warning |
| false-alarm rate | share of on-time tasks wrongly flagged |
| lead time (median, p10) | steps between a correct alert and the deadline |
| monitored tasks, monitored misses | how many tasks the metrics above are computed on |
| excluded: no report before deadline (and how many of them missed) | tasks the trigger could never warn about: they start after their own deadline |
| excluded: outcome unknown | tasks the mission stopped on before their deadline |

A warning one step before the deadline counts in recall but not in timely recall.
`evaluate_tasks(policy, tasks)` does the same on any list of tasks.

Baselines in the same file: `never_alert` and `margin_rule` (alert as soon as the reported estimate
predicts a finish after the deadline).

Evaluating a trained model:

```python
from stable_baselines3 import PPO
from trigger_env import evaluate, margin_rule, never_alert

model = PPO.load("training/saved_models/trigger_ppo_seed0")
print(evaluate(lambda obs: int(model.predict(obs, deterministic=True)[0])))
print(evaluate(margin_rule))
```

### Evaluation script

```bash
python evaluate_trigger.py --models training/saved_models/trigger_ppo_lead*.zip --n-missions 50 --eval-lead-min 5
```

Runs the models (deterministic actions), `always WAIT` and `margin rule` on the same tasks and
saves `evaluation/results/lead_sensitivity_<n>missions_eval_lead<L>.csv` (or `--output`).
`--model` evaluates a single model.

`LEAD_MIN` sweep: `bash run_lead_experiments.sh` (training), then `bash run_lead_evaluation.sh`
(evaluation at horizons 2–20).

Decision threshold (Experiment 3): `bash run_experiment3.sh`. PPO alerts when its ALERT probability
reaches a threshold (0.2–0.8). The threshold with the best timely recall under a false-alarm limit
(`--max-far`, default 0.02) is chosen on `EVAL_MISSIONS[50:100]`, then tested once on
`EVAL_MISSIONS[100:150]` against the default 0.5, the margin rule and always WAIT. No retraining.

---

## Design rules (please keep them when changing the code)

1. **The trigger only observes.** It never changes assignments, queues or robot behaviour.
2. **No future information in observations.** The outcome may shape the reward and the metrics only.
3. **Deadlines are fixed mission data.** Never recompute them from actual start times.
4. **TA-RWARE stays unmodified.** All behaviour is in this repository's rules, so TA-RWARE can be
   updated or replaced.
5. **Training and evaluation use the reserved, disjoint mission sets** (`TRAIN_MISSIONS`, `EVAL_MISSIONS`).

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
- **Optimistic handling estimates** (about 8–11 steps): a better one would account for the picker's queue of commitments.
- **Next:** several seeds per configuration, and paired per-task comparisons with the margin rule.
