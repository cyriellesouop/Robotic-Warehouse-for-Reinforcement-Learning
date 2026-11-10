
#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# Experiment 3: PPO Decision-Threshold Sensitivity
# =====================================================

LEAD_VALUES=(2 5 15)

# Evaluation parameters
EVAL_LEAD_MIN=5
MAX_FAR=0.02

# Validation dataset
VAL_START=50
VAL_MISSIONS=50

# Test dataset
TEST_START=100
TEST_MISSIONS=50

# Directories
MODEL_DIR="training/saved_models"
OUTPUT_DIR="evaluation/experiment3"

mkdir -p "$OUTPUT_DIR"

for lead in "${LEAD_VALUES[@]}"; do

    MODEL="$MODEL_DIR/trigger_ppo_lead${lead}_seed0.zip"

    RUN_OUTPUT="$OUTPUT_DIR/ppo_lead${lead}_seed0"

    if [[ ! -f "$MODEL" ]]; then
        echo "ERROR: Model not found: $MODEL"
        exit 1
    fi

    echo "=========================================="
    echo "Experiment 3: PPO trained with LEAD_MIN=$lead"
    echo "=========================================="

    python experiment3_threshold.py \
        --model "$MODEL" \
        --eval-lead-min "$EVAL_LEAD_MIN" \
        --max-far "$MAX_FAR" \
        --val-start "$VAL_START" \
        --val-missions "$VAL_MISSIONS" \
        --test-start "$TEST_START" \
        --test-missions "$TEST_MISSIONS" \
        --output-dir "$RUN_OUTPUT"

    echo "Completed: LEAD_MIN=$lead"
    echo ""

done

echo "=========================================="
echo "Experiment 3 completed for all models."
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="
