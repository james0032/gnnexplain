#!/bin/bash

# Evaluation wrapper script for GNN Explainer
# This script runs evaluation using the trained model from Kedro pipeline

# Base data directory
DATA_DIR="/projects/aixb/jchung/everycure/influence_estimate/robokop/gnn_ROBOKOP_clean_baseline/data"

# Data files
TRAIN_FILE="${DATA_DIR}/01_raw/robo_train.txt"
VAL_FILE="${DATA_DIR}/01_raw/robo_val.txt"
TEST_FILE="${DATA_DIR}/01_raw/robo_test.txt"
NODE_DICT="${DATA_DIR}/01_raw/node_dict"
REL_DICT="${DATA_DIR}/01_raw/rel_dict"

# Model file
MODEL_PATH="${DATA_DIR}/06_models/trained_model.pkl"

# Output paths
METRICS_SAVE="${DATA_DIR}/07_model_output/evaluation_metrics.pkl"

# Evaluation parameters
BATCH_SIZE=1024

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Trained model not found at $MODEL_PATH"
    echo "Please run training pipeline first: kedro run --pipeline=training"
    exit 1
fi

echo "========================================="
echo "GNN MODEL EVALUATION"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output: $METRICS_SAVE"
echo "========================================="

# Create output directories
mkdir -p "${DATA_DIR}/07_model_output"
mkdir -p "${DATA_DIR}/08_reporting/explanations"

# Run evaluation using Kedro model loader
python src/eval_kedro_model.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --node_dict "$NODE_DICT" \
    --rel_dict "$REL_DICT" \
    --batch_size $BATCH_SIZE \
    --compute_mrr \
    --compute_hits \
    --hit_k_values 1 3 10 \
    --metrics_save_path "$METRICS_SAVE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "EVALUATION COMPLETED SUCCESSFULLY"
    echo "========================================="
    echo "Metrics saved to: $METRICS_SAVE"
    exit 0
else
    echo ""
    echo "========================================="
    echo "EVALUATION FAILED"
    echo "========================================="
    exit 1
fi
