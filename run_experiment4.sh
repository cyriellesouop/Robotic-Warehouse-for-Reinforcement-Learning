
#!/usr/bin/env bash
set -euo pipefail

# ====================================================
# Experiment 4: PPO Reward-Function Sensitivity
# ====================================================

LEAD_MIN=5
TIMESTEPS=200000
N_ENVS=4
SEED=0

# Format:
# Experiment ID | FP penalty | FN penalty

CONFIGS=(
    "R1 2 1"
    "R0 1 1"
    "R2 1 2"
    "R3 2 2"
)

mkdir -p evaluation/experiment4

for config in "${CONFIGS[@]}"; do

    read -r name fp fn <<< "$config"

    echo "======================================="
    echo "Experiment 4: $name"
    echo "FP penalty: $fp"
    echo "FN penalty: $fn"
    echo "LEAD_MIN:   $LEAD_MIN"
    echo "======================================="

    python train_reward_experiment.py \
    --fp-penalty "$fp" \
    --fn-penalty "$fn" \
    --lead-min "$LEAD_MIN" \
    --timesteps "$TIMESTEPS" \
    --n-envs "$N_ENVS" \
    --seed "$SEED" \
        2>&1 | tee \
        "evaluation/experiment4/train_${name}.log"

    echo "Completed: $name"
    echo ""

done

echo "======================================="
echo "Experiment 4 training completed."
echo "======================================="
