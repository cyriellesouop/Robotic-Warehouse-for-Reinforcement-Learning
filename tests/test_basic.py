"""Basic tests. Run from the rl_trigger folder:  python3 -m unittest -v"""

import dataclasses
import unittest

import numpy as np

from dwafp_rl.agent import ANALYZE, N_STATES, WAIT, QTableAgent, discretize
from dwafp_rl.baselines import NeverAnalyze, PersistenceRule
from dwafp_rl.env import ToyFleetEnv
from dwafp_rl.interface import AnalysisOutcome
from dwafp_rl.observation import ObservationBuilder
from dwafp_rl.state import RuntimeSnapshot, TaskView
from dwafp_rl.trigger import GuardConfig, TriggerWrapper


def view(remaining, task_id="T0"):
    # A 40 s task with deadline 50 s, started at t = 0.
    return TaskView(task_id, "robot0", 40.0, 50.0, 0.0, remaining)


class RecordingAnalysis:
    """Stands in for the analysis layer: records requests, always answers 'threatened'."""

    def __init__(self):
        self.requests = []

    def request_analysis(self, snapshot, task_id):
        self.requests.append((snapshot.time, task_id))
        return AnalysisOutcome(threatened=True, repaired=False)


class TestBasic(unittest.TestCase):
    def test_snapshot_is_read_only(self):
        snap = RuntimeSnapshot(0.0, (view(40.0),))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.executing[0].deadline = 100.0  # the trigger cannot move a deadline

    def test_observation_values(self):
        b = ObservationBuilder()
        # t = 10 s, 30 s left: predicted finish 40 s, margin (50 - 40) / 40 = 0.25, progress 0.25.
        obs = b.build(10.0, view(30.0))
        self.assertAlmostEqual(obs[0], 0.25)
        self.assertAlmostEqual(obs[1], 0.25)
        self.assertEqual(obs[2], 0.0)  # first report: no trend
        # t = 12 s, still 30 s left (stuck): predicted finish 42 s, moved 2 s in 2 s -> trend 1.
        obs = b.build(12.0, view(30.0))
        self.assertAlmostEqual(obs[2], 1.0)

    def test_discretize_stays_in_range(self):
        for obs in ([-5, -1, -3], [0, 0, 0], [10, 10, 10]):
            self.assertTrue(0 <= discretize(np.array(obs, dtype=float)) < N_STATES)

    def test_q_update(self):
        agent = QTableAgent(alpha=0.5)
        agent.update(3, ANALYZE, -1.0, None, done=True)    # terminal: Q = 0 + 0.5 (-1 - 0)
        self.assertAlmostEqual(agent.q[3, ANALYZE], -0.5)
        agent.q[7] = [0.2, -0.4]
        agent.update(3, WAIT, 0.0, 7, done=False)          # target = 0 + max(0.2, -0.4)
        self.assertAlmostEqual(agent.q[3, WAIT], 0.1)

    def test_guard_overrules_wait_and_is_logged_separately(self):
        analysis = RecordingAnalysis()
        wrapper = TriggerWrapper(NeverAnalyze(), analysis, GuardConfig(enabled=True, margin=-0.25, cooldown=5.0))
        # Predicted finish 10 + 60 = 70 s, margin (50 - 70) / 40 = -0.5 < -0.25: the guard fires.
        d = wrapper.step(RuntimeSnapshot(10.0, (view(60.0),)))[0]
        self.assertEqual((d.action, d.source), (ANALYZE, "guard"))
        # 2 s later: inside the cooldown, the policy's WAIT stands.
        d = wrapper.step(RuntimeSnapshot(12.0, (view(60.0),)))[0]
        self.assertEqual((d.action, d.source), (WAIT, "policy"))
        self.assertEqual(analysis.requests, [(10.0, "T0")])

    def test_guard_can_be_disabled(self):
        analysis = RecordingAnalysis()
        wrapper = TriggerWrapper(NeverAnalyze(), analysis, GuardConfig(enabled=False))
        d = wrapper.step(RuntimeSnapshot(10.0, (view(60.0),)))[0]
        self.assertEqual(d.action, WAIT)
        self.assertEqual(analysis.requests, [])

    def test_persistence_rule_needs_k_reports(self):
        rule = PersistenceRule(k=3, epsilon=0.05)
        stuck = np.array([0.1, 0.5, 1.0])
        actions = [rule.decide(stuck, view(30.0)) for _ in range(3)]
        self.assertEqual(actions, [WAIT, WAIT, ANALYZE])

    def test_env_is_reproducible(self):
        a, b = ToyFleetEnv(), ToyFleetEnv()
        sa, sb = a.reset(seed=7), b.reset(seed=7)
        self.assertEqual(sa, sb)
        for _ in range(30):
            (sa, fa), (sb, fb) = a.step(), b.step()
            self.assertEqual(sa, sb)
            self.assertEqual(fa, fb)


if __name__ == "__main__":
    unittest.main()
