"""Experiment 4 entry point; uses new functions from train_trigger.py."""
import argparse
from train_trigger import train_reward_experiment


def main():
    parser = argparse.ArgumentParser(description="Experiment 4 reward sensitivity training")
    parser.add_argument("--fp-penalty", type=float, required=True)
    parser.add_argument("--fn-penalty", type=float, required=True)
    parser.add_argument("--lead-min", type=float, default=5)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_reward_experiment(
        fp_penalty=args.fp_penalty,
        fn_penalty=args.fn_penalty,
        lead_min=args.lead_min,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
