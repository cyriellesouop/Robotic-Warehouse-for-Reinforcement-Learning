
#!/usr/bin/env bash

set -euo pipefail

# Experiment 2: LEAD_MIN sensitivity
# All other PPO settings remain unchanged.

LEAD_VALUES=(2 3 5 8 10 15 20)
SEEDS=(0)

TIMESTEPS=200000
N_ENVS=4
ALGO="ppo"

for lead in "${LEAD_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do

        echo "=================================="
        echo "Experiment: LEAD_MIN=$lead"
        echo "Seed: $seed"
        echo "=================================="

        python train_trigger.py \
            --algo "$ALGO" \
            --timesteps "$TIMESTEPS" \
            --n-envs "$N_ENVS" \
            --seed "$seed" \
            --lead-min "$lead"

    done
done

echo "All lead-time experiments completed."
