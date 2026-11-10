"""Tests of TriggerEnv and the mission runner. Run from the repository folder: python -m unittest -v"""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

import tarware_missions as tm
from tarware_missions import Task
from trigger_env import (ALERT, EVAL_MISSIONS, LEAD_MIN, TRAIN_MISSIONS, WAIT, TriggerEnv,
                         evaluate_tasks, features, monitorable, split_missions)


def hand_task(end, deadline=20, n_reports=6, mission_end=100):
    """A 10-step travel task started at 0 with a fixed deadline; its robot reports 9 s left each step."""
    return Task(agv=0, phase=0, cell=(0, 0), loc_id=1, p=10.0, deadline=deadline, start=0, end=end,
                mission_end=mission_end, trace=[(t, 9.0) for t in range(1, n_reports + 1)])


def play(task, actions):
    """Serves `task` through TriggerEnv with the given actions; returns the rewards and last info."""
    env = TriggerEnv()
    env.pending = [task]              # serve this task instead of generating a mission
    env.reset()
    rewards, info = [], {}
    for a in actions:
        _, r, done, _, info = env.step(a)
        rewards.append(r)
        if done:
            break
    return rewards, info


class TestMissions(unittest.TestCase):
    def test_deadlines_are_fixed_by_the_reference_plan(self):
        # d = reference finish x (1 + SLACK): the sum of the nominal durations of the AGV's queue
        # up to the task, whatever happened during execution (late starts never move it).
        tasks = tm.run_mission(3)
        for a in range(3):
            q = [t for t in tasks if t.agv == a]
            ref_finish = np.cumsum([t.p for t in q])
            self.assertEqual([t.deadline for t in q], [round(f * (1 + tm.SLACK)) for f in ref_finish])

    def test_reports_only_while_executing_and_before_the_deadline(self):
        for t in tm.run_mission(4):
            times = [s for s, _ in t.trace]
            self.assertEqual(times, sorted(set(times)))           # one report per step
            for s in times:
                self.assertTrue(t.start <= s < t.deadline)
                if t.end is not None:
                    self.assertTrue(s < t.end)                    # nothing after completion

    def test_travel_estimate_uses_the_nominal_timing_model(self):
        # Nominal travel = TURN_FACTOR x path length, so the estimate is TURN_FACTOR x cells left.
        for t in tm.run_mission(5):
            if t.kind == "travel":
                for _, r in t.trace:
                    self.assertAlmostEqual(r / tm.TURN_FACTOR, round(r / tm.TURN_FACTOR), places=6)

    def test_missions_complete(self):
        # The fixed rules must not deadlock (every task completes within the episode).
        for seed in range(5):
            self.assertTrue(all(t.end is not None for t in tm.run_mission(seed)))

    def test_outcome_of_an_unfinished_task(self):
        self.assertFalse(hand_task(end=15).missed)                  # completed by its deadline
        self.assertTrue(hand_task(end=21).missed)                   # completed after it
        self.assertTrue(hand_task(end=None, mission_end=25).missed)   # deadline passed, never completed
        self.assertIsNone(hand_task(end=None, mission_end=10).missed)  # mission stopped before d: unknown


class TestTriggerEnv(unittest.TestCase):
    def test_gymnasium_interface(self):
        check_env(TriggerEnv(range(0, 1000)), skip_render_check=True)

    def test_early_alert_on_a_missed_task(self):
        rewards, info = play(hand_task(end=None), [WAIT, ALERT])  # alert at t = 2, lead 18 >= LEAD_MIN
        self.assertEqual(rewards, [0.0, 1.0])
        self.assertEqual((info["alert"], info["missed"], info["lead"]), (True, True, 18))

    def test_late_alert_earns_less(self):
        task = hand_task(end=None, deadline=8)                    # alert at t = 6: lead 2
        rewards, _ = play(task, [WAIT] * 5 + [ALERT])
        self.assertAlmostEqual(rewards[-1], 2 / LEAD_MIN)

    def test_false_alarm(self):
        rewards, info = play(hand_task(end=15), [ALERT])          # finishes at 15 <= 20: on time
        self.assertEqual(rewards, [-1.0])
        self.assertFalse(info["missed"])

    def test_missed_warning_is_penalized_once(self):
        rewards, info = play(hand_task(end=25), [WAIT] * 10)       # finishes late, never warned
        self.assertEqual(rewards, [0.0] * 5 + [-1.0])             # 6 reports, penalty at the end only
        self.assertEqual((info["alert"], info["missed"]), (False, True))

    def test_never_completed_task_cannot_escape_the_penalty(self):
        rewards, _ = play(hand_task(end=None), [WAIT] * 10)
        self.assertEqual(rewards[-1], -1.0)

    def test_quiet_on_an_on_time_task(self):
        rewards, _ = play(hand_task(end=15), [WAIT] * 10)
        self.assertEqual(sum(rewards), 0.0)

    def test_tasks_that_cannot_be_judged_or_warned_are_not_monitored(self):
        self.assertFalse(monitorable(hand_task(end=None, mission_end=10)))   # outcome unknown
        self.assertFalse(monitorable(hand_task(end=None, n_reports=0)))      # no report before d
        self.assertTrue(monitorable(hand_task(end=15)))

    def test_observation_never_reveals_the_outcome(self):
        on_time, late = hand_task(end=15), hand_task(end=None)    # same reports, different outcomes
        for k in range(len(on_time.trace)):
            np.testing.assert_array_equal(features(on_time, k), features(late, k))

    def test_same_missions_same_tasks(self):
        a, b = TriggerEnv(range(7, 100)), TriggerEnv(range(7, 100))
        for _ in range(20):
            np.testing.assert_array_equal(a.reset()[0], b.reset()[0])

    def test_training_and_evaluation_missions_never_overlap(self):
        self.assertEqual(len(set(TRAIN_MISSIONS) & set(EVAL_MISSIONS)), 0)
        for n in (1, 4, 11, 16):                                   # parallel training copies
            slices = split_missions(TRAIN_MISSIONS, n)
            seen = set()
            for s in slices:
                self.assertTrue(TRAIN_MISSIONS.start <= s.start and s.stop <= TRAIN_MISSIONS.stop)
                self.assertTrue(seen.isdisjoint(s))
                seen.update(s)

    def test_evaluation_metrics(self):
        # Policy: ALERT once 60 % of the nominal duration has elapsed (report at t = 6).
        policy = lambda obs: ALERT if obs[1] >= 0.6 else WAIT
        tasks = [hand_task(end=None),                              # missed, alert at 6, lead 14: timely
                 hand_task(end=None, deadline=8, n_reports=7),     # missed, alert at 6, lead 2: late
                 hand_task(end=15),                                # on time, alert at 6: false alarm
                 hand_task(end=None, n_reports=0, mission_end=30), # no report before d, missed
                 hand_task(end=None, mission_end=3)]               # outcome unknown
        m = evaluate_tasks(policy, tasks)
        self.assertEqual(m["monitored tasks"], 3)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertEqual(m["recall"], 1.0)                         # both misses warned...
        self.assertEqual(m["timely recall"], 0.5)                  # ...but only one early enough
        self.assertEqual(m["false-alarm rate"], 1.0)
        self.assertEqual(m["lead time median"], 8.0)
        self.assertAlmostEqual(m["mean reward per task"], (1.0 + 2 / LEAD_MIN - 1.0) / 3)  # +1, +0.4, -1
        self.assertEqual(m["excluded: no report before deadline"], 1)
        self.assertEqual(m["  of which missed"], 1)
        self.assertEqual(m["excluded: outcome unknown"], 1)


if __name__ == "__main__":
    unittest.main()
