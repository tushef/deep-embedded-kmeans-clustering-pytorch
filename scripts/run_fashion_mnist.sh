#!/bin/bash

# Fashion-MNIST Example Run Script
# This script demonstrates how to run DEKM on Fashion-MNIST dataset

echo "Running DEKM on Fashion-MNIST dataset with 10 clusters..."

python main.py \
    --dataset fashionmnist \
    --n-clusters 10 \
    --batch-size 256 \
    --pretrain-epochs 200 \
    --clustering-epochs 200 \
    --pretrain-lr 0.001 \
    --clustering-lr 0.0001 \
    --embedding-size 12 \
    --save-dir pretrained_weights \
    --results-dir results \
    --verbose

echo "Training completed! Check results/ directory for outputs."
