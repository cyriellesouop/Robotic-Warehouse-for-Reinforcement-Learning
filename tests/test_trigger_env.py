"""Tests of TriggerEnv and the mission runner. Run from rl_trigger/: python -m unittest -v"""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

import tarware_missions as tm
from tarware_missions import Task
from trigger_env import ALERT, LEAD_MIN, WAIT, TriggerEnv, features


def hand_task(end, deadline=20, n_reports=6):
    """A 10-step task started at 0 with a fixed deadline; its robot reports 9 s left each step."""
    return Task(agv=0, kind="travel", cell=(0, 0), loc_id=1, p=10.0, deadline=deadline, start=0,
                end=end, trace=[(t, 9.0) for t in range(1, n_reports + 1)])


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

    def test_missions_complete(self):
        # The fixed rules must not deadlock (every task completes within the episode).
        for seed in range(5):
            self.assertTrue(all(t.end is not None for t in tm.run_mission(seed)))


class TestTriggerEnv(unittest.TestCase):
    def test_gymnasium_interface(self):
        check_env(TriggerEnv(seed=0), skip_render_check=True)

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

    def test_observation_never_reveals_the_outcome(self):
        on_time, late = hand_task(end=15), hand_task(end=None)    # same reports, different outcomes
        for k in range(len(on_time.trace)):
            np.testing.assert_array_equal(features(on_time, k), features(late, k))

    def test_same_seed_same_tasks(self):
        a, b = TriggerEnv(seed=7), TriggerEnv(seed=7)
        for _ in range(20):
            np.testing.assert_array_equal(a.reset()[0], b.reset()[0])


if __name__ == "__main__":
    unittest.main()
