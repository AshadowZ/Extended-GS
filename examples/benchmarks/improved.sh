#!/bin/bash

# Benchmark script for comparing DefaultStrategy vs ImprovedStrategy with different budgets
# on Mip-NeRF360 dataset
# Usage: ./examples/benchmarks/improved.sh

SCENE_DIR="data/360_v2"
BASE_RESULT_DIR="output/benchmark_improved_comparison"
SCENE_LIST="garden" # Only benchmark the garden scene; others: bicycle stump bonsai counter kitchen room
RENDER_TRAJ_PATH="ellipse"

# Create results directory
mkdir -p $BASE_RESULT_DIR

echo "=== Strategy Comparison Benchmark ==="
echo "Dataset: $SCENE_DIR"
echo "Scenes: $SCENE_LIST"
echo "Strategies: default, improved_1M, improved_2M"
echo

# Function to run training and evaluation for a specific strategy
run_strategy() {
    local STRATEGY_NAME=$1
    local STRATEGY_TYPE=$2
    local BUDGET=$3
    local RESULT_DIR="$BASE_RESULT_DIR/$STRATEGY_NAME"

    echo "=== Running $STRATEGY_NAME (budget: $BUDGET) ==="

    for SCENE in $SCENE_LIST;
    do
        if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
            DATA_FACTOR=2
        else
            DATA_FACTOR=4
        fi

        echo "Running $SCENE with $STRATEGY_NAME (data factor: $DATA_FACTOR)"

        # Build command based on strategy type
        if [ "$STRATEGY_TYPE" = "default" ]; then
            # Default strategy doesn't support budget parameter
            TRAIN_CMD="CUDA_VISIBLE_DEVICES=0 python examples/extended_trainer.py $STRATEGY_TYPE --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
                --render_traj_path $RENDER_TRAJ_PATH \
                --data_dir $SCENE_DIR/$SCENE/ \
                --result_dir $RESULT_DIR/$SCENE/"

            EVAL_CMD="CUDA_VISIBLE_DEVICES=0 python examples/extended_trainer.py $STRATEGY_TYPE --disable_viewer --data_factor $DATA_FACTOR \
                --render_traj_path $RENDER_TRAJ_PATH \
                --data_dir $SCENE_DIR/$SCENE/ \
                --result_dir $RESULT_DIR/$SCENE/ \
                --ckpt \$CKPT"
        else
            # Improved strategy supports budget parameter
            TRAIN_CMD="CUDA_VISIBLE_DEVICES=0 python examples/extended_trainer.py $STRATEGY_TYPE --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
                --strategy.budget $BUDGET \
                --render_traj_path $RENDER_TRAJ_PATH \
                --data_dir $SCENE_DIR/$SCENE/ \
                --result_dir $RESULT_DIR/$SCENE/"

            EVAL_CMD="CUDA_VISIBLE_DEVICES=0 python examples/extended_trainer.py $STRATEGY_TYPE --disable_viewer --data_factor $DATA_FACTOR \
                --strategy.budget $BUDGET \
                --render_traj_path $RENDER_TRAJ_PATH \
                --data_dir $SCENE_DIR/$SCENE/ \
                --result_dir $RESULT_DIR/$SCENE/ \
                --ckpt \$CKPT"
        fi

        # train without eval
        eval $TRAIN_CMD

        # run eval and render
        for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
        do
            eval $EVAL_CMD
        done
    done
}

# Run all strategies
run_strategy "default" "default" ""
run_strategy "improved_1M" "improved" "1000000"
run_strategy "improved_2M" "improved" "2000000"

# Generate comparison tables
echo "=== Generating Comparison Tables ==="

# Function to extract metrics from JSON files for specific step
extract_metrics_for_step() {
    local SCENE=$1
    local STRATEGY=$2
    local STEP=$3
    local RESULT_DIR="$BASE_RESULT_DIR/$STRATEGY/$SCENE"

    # Find the eval stats file for the specific step
    local STEP_STATS=$(ls $RESULT_DIR/stats/val*_step${STEP}.json 2>/dev/null)
    if [ -z "$STEP_STATS" ]; then
        # If exact match not found, try pattern matching
        STEP_STATS=$(ls $RESULT_DIR/stats/val*.json | grep "_step${STEP}" | tail -1)
    fi
    if [ -z "$STEP_STATS" ]; then
        # If specific step not found, use the latest
        STEP_STATS=$(ls $RESULT_DIR/stats/val*.json | tail -1)
    fi

    if [ -f "$STEP_STATS" ]; then
        # Extract metrics from JSON
        local PSNR=$(python -c "import json; data=json.load(open('$STEP_STATS')); print(data.get('psnr', 'N/A'))")
        local SSIM=$(python -c "import json; data=json.load(open('$STEP_STATS')); print(data.get('ssim', 'N/A'))")
        local LPIPS=$(python -c "import json; data=json.load(open('$STEP_STATS')); print(data.get('lpips', 'N/A'))")
        local NUM_GS=$(python -c "import json; data=json.load(open('$STEP_STATS')); print(data.get('num_GS', 'N/A'))")

        # Find training time from train stats for the specific step
        local TRAIN_STATS=$(ls $RESULT_DIR/stats/train*_step${STEP}_rank0.json 2>/dev/null)
        if [ -z "$TRAIN_STATS" ]; then
            # If exact match not found, try pattern matching
            TRAIN_STATS=$(ls $RESULT_DIR/stats/train*_rank0.json | grep "_step${STEP}" | tail -1)
        fi
        if [ -z "$TRAIN_STATS" ]; then
            # If specific step not found, use the latest
            TRAIN_STATS=$(ls $RESULT_DIR/stats/train*_rank0.json | tail -1)
        fi

        local TRAINING_TIME="N/A"
        if [ -f "$TRAIN_STATS" ]; then
            TRAINING_TIME=$(python -c "import json; data=json.load(open('$TRAIN_STATS')); print(data.get('ellipse_time', 'N/A'))")
        fi

        echo "$STRATEGY|$PSNR|$SSIM|$LPIPS|$NUM_GS|$TRAINING_TIME"
    else
        echo "$STRATEGY|N/A|N/A|N/A|N/A|N/A"
    fi
}

# Generate markdown summary
SUMMARY_FILE="$BASE_RESULT_DIR/comparison_summary.md"
echo "# Strategy Comparison Results on Mip-NeRF360 Dataset" > $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "This document compares the performance of DefaultStrategy vs ImprovedStrategy with different budgets (1M and 2M) across all Mip-NeRF360 scenes." >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Arrays to store metrics for averaging at 30k steps
declare -A PSNR_VALUES_30K
declare -A SSIM_VALUES_30K
declare -A LPIPS_VALUES_30K
declare -A NUM_GS_VALUES_30K
declare -A TRAINING_TIME_VALUES_30K

# Initialize arrays
for STRATEGY in "default" "improved_1M" "improved_2M"; do
    PSNR_VALUES_30K[$STRATEGY]=""
    SSIM_VALUES_30K[$STRATEGY]=""
    LPIPS_VALUES_30K[$STRATEGY]=""
    NUM_GS_VALUES_30K[$STRATEGY]=""
    TRAINING_TIME_VALUES_30K[$STRATEGY]=""
done

for SCENE in $SCENE_LIST;
do
    echo "## Scene: $SCENE" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE

    # Table for 7k iterations
    echo "### At 7,000 Iterations" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    echo "| Strategy | PSNR | SSIM | LPIPS | Num GS | Training Time |" >> $SUMMARY_FILE
    echo "|----------|------|------|-------|--------|---------------|" >> $SUMMARY_FILE

    # Extract metrics for each strategy at 7k
    DEFAULT_METRICS_7K=$(extract_metrics_for_step $SCENE "default" "6999")
    IMPROVED_1M_METRICS_7K=$(extract_metrics_for_step $SCENE "improved_1M" "6999")
    IMPROVED_2M_METRICS_7K=$(extract_metrics_for_step $SCENE "improved_2M" "6999")

    echo "| $DEFAULT_METRICS_7K |" >> $SUMMARY_FILE
    echo "| $IMPROVED_1M_METRICS_7K |" >> $SUMMARY_FILE
    echo "| $IMPROVED_2M_METRICS_7K |" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE

    # Table for 30k iterations
    echo "### At 30,000 Iterations" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    echo "| Strategy | PSNR | SSIM | LPIPS | Num GS | Training Time |" >> $SUMMARY_FILE
    echo "|----------|------|------|-------|--------|---------------|" >> $SUMMARY_FILE

    # Extract metrics for each strategy at 30k
    DEFAULT_METRICS_30K=$(extract_metrics_for_step $SCENE "default" "29999")
    IMPROVED_1M_METRICS_30K=$(extract_metrics_for_step $SCENE "improved_1M" "29999")
    IMPROVED_2M_METRICS_30K=$(extract_metrics_for_step $SCENE "improved_2M" "29999")

    echo "| $DEFAULT_METRICS_30K |" >> $SUMMARY_FILE
    echo "| $IMPROVED_1M_METRICS_30K |" >> $SUMMARY_FILE
    echo "| $IMPROVED_2M_METRICS_30K |" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE

    # Parse metrics and add to arrays for averaging (30k only for summary)
    parse_and_add_metrics() {
        local METRICS=$1
        local STRATEGY=$2

        IFS='|' read -r STRAT PSNR SSIM LPIPS NUM_GS TIME <<< "$METRICS"

        # Only add numeric values
        if [[ $PSNR =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            PSNR_VALUES_30K[$STRATEGY]="${PSNR_VALUES_30K[$STRATEGY]} $PSNR"
        fi
        if [[ $SSIM =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            SSIM_VALUES_30K[$STRATEGY]="${SSIM_VALUES_30K[$STRATEGY]} $SSIM"
        fi
        if [[ $LPIPS =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            LPIPS_VALUES_30K[$STRATEGY]="${LPIPS_VALUES_30K[$STRATEGY]} $LPIPS"
        fi
        if [[ $NUM_GS =~ ^[0-9]+$ ]]; then
            NUM_GS_VALUES_30K[$STRATEGY]="${NUM_GS_VALUES_30K[$STRATEGY]} $NUM_GS"
        fi
        if [[ $TIME =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            TRAINING_TIME_VALUES_30K[$STRATEGY]="${TRAINING_TIME_VALUES_30K[$STRATEGY]} $TIME"
        fi
    }

    parse_and_add_metrics "$DEFAULT_METRICS_30K" "default"
    parse_and_add_metrics "$IMPROVED_1M_METRICS_30K" "improved_1M"
    parse_and_add_metrics "$IMPROVED_2M_METRICS_30K" "improved_2M"
done

# Calculate averages
calculate_average() {
    local VALUES=($1)
    local SUM=0
    local COUNT=0

    for VALUE in "${VALUES[@]}"; do
        if [[ $VALUE =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            SUM=$(echo "$SUM + $VALUE" | bc -l)
            COUNT=$((COUNT + 1))
        fi
    done

    if [ $COUNT -gt 0 ]; then
        echo "scale=3; $SUM / $COUNT" | bc -l
    else
        echo "N/A"
    fi
}

# Generate overall summary table (based on 30k iterations)
echo "## Overall Summary (At 30,000 Iterations)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "| Strategy | Avg PSNR | Avg SSIM | Avg LPIPS | Avg Num GS | Avg Training Time |" >> $SUMMARY_FILE
echo "|----------|----------|----------|-----------|------------|------------------|" >> $SUMMARY_FILE

for STRATEGY in "default" "improved_1M" "improved_2M"; do
    AVG_PSNR=$(calculate_average "${PSNR_VALUES_30K[$STRATEGY]}")
    AVG_SSIM=$(calculate_average "${SSIM_VALUES_30K[$STRATEGY]}")
    AVG_LPIPS=$(calculate_average "${LPIPS_VALUES_30K[$STRATEGY]}")
    AVG_NUM_GS=$(calculate_average "${NUM_GS_VALUES_30K[$STRATEGY]}")
    AVG_TIME=$(calculate_average "${TRAINING_TIME_VALUES_30K[$STRATEGY]}")

    echo "| $STRATEGY | $AVG_PSNR | $AVG_SSIM | $AVG_LPIPS | $AVG_NUM_GS | $AVG_TIME |" >> $SUMMARY_FILE
done

echo "" >> $SUMMARY_FILE
echo "*Note: Averages are calculated across all completed scenes*" >> $SUMMARY_FILE

echo "=== Comparison Complete ==="
echo "Results saved to: $SUMMARY_FILE"
