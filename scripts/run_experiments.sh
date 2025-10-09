#!/bin/bash

# Comprehensive Experiments Script
# This script runs multiple experiments with different configurations

echo "Running comprehensive DEKM experiments..."

# Experiment 1: MNIST with different cluster numbers
echo "Experiment 1: MNIST with 5 clusters"
python main.py --dataset mnist --n-clusters 5 --save-dir pretrained_weights --results-dir results --verbose

echo "Experiment 2: MNIST with 10 clusters"
python main.py --dataset mnist --n-clusters 10 --save-dir pretrained_weights --results-dir results --verbose

echo "Experiment 3: MNIST with 15 clusters"
python main.py --dataset mnist --n-clusters 15 --save-dir pretrained_weights --results-dir results --verbose

# Experiment 4: Fashion-MNIST
echo "Experiment 4: Fashion-MNIST with 10 clusters"
python main.py --dataset fashionmnist --n-clusters 10 --save-dir pretrained_weights --results-dir results --verbose

# Experiment 5: CIFAR-10
echo "Experiment 5: CIFAR-10 with 10 clusters"
python main.py --dataset cifar10 --n-clusters 10 --embedding-size 16 --save-dir pretrained_weights --results-dir results --verbose

echo "All experiments completed! Check results/ directory for outputs."
