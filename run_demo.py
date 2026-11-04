"""Train the Q-learning trigger in the toy simulator, then compare it with the baselines.

Run from this folder:  python3 run_demo.py
"""

from dwafp_rl.agent import QTableAgent
from dwafp_rl.baselines import MarginRule, NeverAnalyze, PersistenceRule
from dwafp_rl.env import ToyFleetEnv
from dwafp_rl.evaluate import evaluate, print_table
from dwafp_rl.train import RewardConfig, run_episode, train
from dwafp_rl.trigger import GuardConfig, TriggerWrapper, write_log_csv


def main() -> None:
    agent = QTableAgent(seed=0)
    history = train(agent, episodes=3000)
    first, last = history[:300], history[-300:]
    print(f"training reward per task: first 300 missions {sum(h.reward for h in first) / (4 * 300):.3f}, "
          f"last 300 missions {sum(h.reward for h in last) / (4 * 300):.3f}\n")

    # Same 500 evaluation missions for every policy (paired comparison). Guard off, except in
    # the last row, where it is the deployment configuration.
    results = {
        "never analyze (reactive)": evaluate(NeverAnalyze()),
        "margin rule (margin < 0)": evaluate(MarginRule(0.0)),
        "persistence rule (k=3)": evaluate(PersistenceRule(k=3)),
        "Q-learning": evaluate(agent),
        "Q-learning + guard": evaluate(agent, guard=GuardConfig(enabled=True)),
    }
    print_table(results)

    # Decision log of one mission (every task, every report, with the decision's source).
    env = ToyFleetEnv()
    wrapper = TriggerWrapper(agent, env, GuardConfig(enabled=True))
    run_episode(env, wrapper, RewardConfig(), seed=100_000)
    write_log_csv(wrapper.log, "decisions_example.csv")
    print("\ndecision log of one mission written to decisions_example.csv")


if __name__ == "__main__":
    main()
