#!/bin/bash

# CIFAR-10 Example Run Script
# This script demonstrates how to run DEKM on CIFAR-10 dataset

echo "Running DEKM on CIFAR-10 dataset with 10 clusters..."

python main.py \
    --dataset cifar10 \
    --n-clusters 10 \
    --batch-size 256 \
    --pretrain-epochs 300 \
    --clustering-epochs 250 \
    --pretrain-lr 0.001 \
    --clustering-lr 0.0001 \
    --embedding-size 16 \
    --autoencoder-layers 64 128 256 \
    --save-dir pretrained_weights \
    --results-dir results \
    --verbose

echo "Training completed! Check results/ directory for outputs."
