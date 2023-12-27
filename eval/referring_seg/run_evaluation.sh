#!/bin/sh

## USAGE

## bash eval/referring_seg/run_evaluation.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE


# Adjust the environment variable if you have multiple gpus available, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 if you have 4 GPUs available
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=24999

# Positional arguments for the bash scripts
CKPT_PATH=$1
RESULT_PATH=$2

# RefCOCO
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|val" --results_path "$RESULT_PATH" --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testA" --results_path "$RESULT_PATH" --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testB" --results_path "$RESULT_PATH" --pretrained

# RefCOCO+
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|val" --results_path "$RESULT_PATH" --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testA" --results_path "$RESULT_PATH" --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testB" --results_path "$RESULT_PATH" --pretrained

# RefCOCOg
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcocog|val" --results_path "$RESULT_PATH" --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcocog|test" --results_path "$RESULT_PATH" --pretrained
