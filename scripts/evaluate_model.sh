#!/bin/bash

# Model Evaluation Script
# This script evaluates a trained DEKM model and generates visualizations

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path> [dataset] [output_dir]"
    echo "Example: $0 pretrained_weights/mnist_dekm.pth mnist plots"
    exit 1
fi

MODEL_PATH=$1
DATASET=${2:-mnist}
OUTPUT_DIR=${3:-plots}

echo "Evaluating model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"

python evaluate.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --verbose

echo "Evaluation completed! Check $OUTPUT_DIR for results."
