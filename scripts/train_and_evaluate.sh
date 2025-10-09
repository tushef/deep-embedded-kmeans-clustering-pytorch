#!/bin/bash

# Complete Training and Evaluation Pipeline
# This script trains a model and then evaluates it

DATASET=${1:-mnist}
N_CLUSTERS=${2:-10}

echo "Training DEKM model on $DATASET with $N_CLUSTERS clusters..."

# Train the model
python main.py \
    --dataset "$DATASET" \
    --n-clusters "$N_CLUSTERS" \
    --batch-size 256 \
    --pretrain-epochs 200 \
    --clustering-epochs 200 \
    --pretrain-lr 0.001 \
    --clustering-lr 0.0001 \
    --embedding-size 10 \
    --save-dir "pretrained_weights/${DATASET}_dekm.pth" \
    --results-dir "results" \
    --verbose

echo "Training completed! Now evaluating the model..."

# Evaluate the model
python evaluate.py \
    --model-path "pretrained_weights/${DATASET}_dekm.pth" \
    --dataset "$DATASET" \
    --output-dir "plots/${DATASET}_evaluation" \
    --verbose

echo "Training and evaluation pipeline completed!"
echo "Check plots/${DATASET}_evaluation/ for visualization results."
