#!/bin/sh

## USAGE

## bash eval/gcg/run_evaluation.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE


export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=24999
NUM_GPUS=1  # Adjust it as per the available #GPU

# Positional arguments for the bash scripts
CKPT_PATH=$1
RESULT_PATH=$2

# Path to the GranD-f evaluation dataset images directory
IMAGE_DIR=./data/GranDf/GranDf_HA_images/val_test

# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/gcg/infer.py --hf_model_path "$CKPT_PATH" --img_dir "$IMAGE_DIR" --output_dir "$RESULT_PATH"

# Path to the GranD-f evaluation dataset ground-truths directory
GT_DIR=./data/GranDf/annotations/val_test

# Evaluate
python eval/gcg/evaluate.py --prediction_dir_path "$RESULT_PATH" --gt_dir_path "$GT_DIR" --split "val"
python eval/gcg/evaluate.py --prediction_dir_path "$RESULT_PATH" --gt_dir_path "$GT_DIR" --split "test"
