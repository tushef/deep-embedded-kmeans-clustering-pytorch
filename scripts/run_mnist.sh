#!/bin/bash

# MNIST Example Run Script
# This script demonstrates how to run DEKM on MNIST dataset

echo "Running DEKM on MNIST dataset with 10 clusters..."

python main.py \
    --dataset mnist \
    --n-clusters 10 \
    --batch-size 256 \
    --pretrain-epochs 200 \
    --clustering-epochs 200 \
    --pretrain-lr 0.001 \
    --clustering-lr 0.0001 \
    --embedding-size 10 \
    --save-dir pretrained_weights \
    --results-dir results \
    --verbose

echo "Training completed! Check results/ directory for outputs."
