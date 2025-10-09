@echo off

REM Quick Test Script for Windows
REM This script runs a quick test with fewer epochs for testing purposes

echo Running quick DEKM test on MNIST with reduced epochs...

python main.py ^
    --dataset mnist ^
    --n-clusters 10 ^
    --batch-size 512 ^
    --pretrain-epochs 50 ^
    --clustering-epochs 50 ^
    --pretrain-lr 0.001 ^
    --clustering-lr 0.0001 ^
    --embedding-size 10 ^
    --save-dir pretrained_weights ^
    --results-dir results ^
    --verbose

echo Quick test completed!
pause
