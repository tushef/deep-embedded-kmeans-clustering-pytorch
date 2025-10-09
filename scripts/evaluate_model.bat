@echo off

REM Model Evaluation Script for Windows
REM This script evaluates a trained DEKM model and generates visualizations

if "%1"=="" (
    echo Usage: %0 ^<model_path^> [dataset] [output_dir]
    echo Example: %0 pretrained_weights\mnist_dekm.pth mnist plots
    exit /b 1
)

set MODEL_PATH=%1
set DATASET=%2
if "%DATASET%"=="" set DATASET=mnist

set OUTPUT_DIR=%3
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=plots

echo Evaluating model: %MODEL_PATH%
echo Dataset: %DATASET%
echo Output directory: %OUTPUT_DIR%

python evaluate.py ^
    --model-path "%MODEL_PATH%" ^
    --dataset "%DATASET%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --verbose

echo Evaluation completed! Check %OUTPUT_DIR% for results.
pause
