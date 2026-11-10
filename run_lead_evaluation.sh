
#!/usr/bin/env bash
set -euo pipefail

# Experiment 2: LEAD_MIN Sensitivity Evaluation
# Evaluate all trained PPO models on the same held-out missions.
# No retraining is performed.

N_MISSIONS=50

# Common evaluation lead-time thresholds
EVAL_LEAD_VALUES=(2 3 5 8 10 15 20)

# Directory containing trained models
MODEL_DIR="training/saved_models"

# Output directory
RESULTS_DIR="evaluation/results"
mkdir -p "$RESULTS_DIR"

# Find all models trained with different LEAD_MIN values
shopt -s nullglob
MODELS=("$MODEL_DIR"/trigger_ppo_lead*_seed*.zip)
shopt -u nullglob

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No trained PPO lead-time models found."
    exit 1
fi

echo "=========================================="
echo "Experiment 2: LEAD_MIN Sensitivity"
echo "=========================================="
echo "Number of missions: $N_MISSIONS"
echo "Number of models: ${#MODELS[@]}"
echo "Evaluation thresholds: ${EVAL_LEAD_VALUES[*]}"
echo "=========================================="

# Evaluate all models at each common threshold
for eval_lead in "${EVAL_LEAD_VALUES[@]}"; do

    echo ""
    echo "=========================================="
    echo "Evaluation LEAD_MIN = $eval_lead"
    echo "=========================================="

    python evaluate_trigger.py \
        --models "${MODELS[@]}" \
        --n-missions "$N_MISSIONS" \
        --eval-lead-min "$eval_lead" \
        --output "$RESULTS_DIR/lead_sensitivity_${N_MISSIONS}missions_eval_lead${eval_lead}.csv"

done

echo ""
echo "=========================================="
echo "All evaluations completed successfully."
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="
