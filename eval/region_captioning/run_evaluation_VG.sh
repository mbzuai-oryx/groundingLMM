#!/bin/sh

## USAGE

## bash eval/region_captioning/run_evaluation.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE


export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=24999
NUM_GPUS=1  # Adjust it as per the available #GPU

# Positional arguments for the bash scripts
CKPT_PATH=$1
RESULT_PATH=$2

# Adjust if needed
ANNOTATION_FILE=./data/visual_genome/test_caption.json
IMAGE_DIR=./data/visual_genome/images
DATASET=vg

# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/region_captioning/infer.py --hf_model_path "$CKPT_PATH" --annotation_file "$ANNOTATION_FILE" --image_dir "$IMAGE_DIR" --dataset "$DATASET" --results_dir "$RESULT_PATH"


# Evaluate
python eval/region_captioning/evaluate.py --annotation_file "$ANNOTATION_FILE" --results_dir "$RESULT_PATH"
