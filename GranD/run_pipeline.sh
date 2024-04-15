#!/bin/bash

# Exit on error, uninitialized var, and ensure commands in pipes are all checked for success
set -euo pipefail

# Input arguments - Image directory path, output predictions directory path, checkpoints directory path containing all checkpoints and directory containing original SAM annotation files
IMG_DIR=$1
PRED_DIR=$2
CKPT_DIR=$3
SAM_ANNOTATIONS_DIR=$4

# Adjust below configuration as per your setup
NUM_GPUs=1
GPU_IDs="0"
MASTER_PORT=1342


# NOTE: The pipeline contains multiple models from different open-source resources. The dependencies to run varies from one model to other. That's why, we had to create almost 10 different conda environments with different dependencies to run the complete pipeline. Please follow the instructions at the corresponding model directory to install the dependencies. We will welcome any pull request to make this process easy. Thank You.


# We define some commands below to activate the correct conda environments
run_in_env() {
    local env="$1"
    shift
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$env"
    "$@"
}

run_in_env_targeted() {
    local env="$1"
    shift
    export CUDA_VISIBLE_DEVICES=$GPU_IDsS
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$env"
    "$@"
}

# NOTE: Here we assume to have ten conda environments created, namely grand_env_1, grand_env_2, ---, grand_env_9 and grand_env_utils. The requirements for these environments is available under environments directory.


# 1. Landmark
run_in_env grand_env_1 pushd level_1_inference/1_landmark_categorization
    python infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --gpu_ids "$GPU_IDsS" --llava_model_path "$CKPT_DIR/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
popd

# 2. Depth Maps
run_in_env_targeted grand_env_2 level_1_inference/pushd 2_depth_maps
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --model_weights "$CKPT_DIR/dpt_beit_large_512.pt"
popd

# 3. Image Tagging
run_in_env_targeted grand_env_3 pushd level_1_inference/3_image_tagging
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --model-type tag2text --checkpoint "$CKPT_DIR/tag2text_swin_14m.pth"

    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --model-type ram --checkpoint "$CKPT_DIR/ram_swin_large_14m.pth"
popd

# 4. Object Detection using Co-DETR
run_in_env grand_env_1 pushd level_1_inference/4_co_detr
    python launch_codetr_multi_gpu_inference.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --ckpt_path "$CKPT_DIR/co_deformable_detr_swin_large_900q_3x_coco.pth" --gpu_ids "$GPU_IDs"
popd

# 5. Object Detection using EVA-02
run_in_env_targeted grand_env_4 pushd level_1_inference/5_eva_02
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --model_name 'eva-02-01'
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --model_name 'eva-02-02'
popd

# 6. Open Vocabulary Detection using OWL-ViT
run_in_env grand_env_1 pushd level_1_inference/6_owl_vit
    python launch_owl_vit_multi_gpu_inference.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --tags_dir_path "$PRED_DIR" --gpu_ids "$GPU_IDs"
popd

# 7. Open Vocabulary Detection using POMP
run_in_env grand_env_4 pushd level_1_inference/7_pomp
    python launch_pomp_multi_gpu_inference.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --tags_dir_path "$PRED_DIR" --gpu_ids "$GPU_IDs"
popd

# 8. Attribute Detection and Grounding using GRIT
run_in_env_targeted grand_env_3 level_1_inference/pushd 8_grit \
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR"
popd

# 9. Open Vocabulary Classification using OV-SAM
run_in_env grand_env_5 pushd level_1_inference/9_ov_sam
    python launch_ov_sam_multi_gpu_inference.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --sam_annotations_dir "$SAM_ANNOTATIONS_DIR" --gpu_ids "$GPU_IDs"
popd

# 10. Generate Level-1 Scene Graph
run_in_env grand_env_utils
    python utils/merge_json_level_1_with_nms.py --image_dir_path "$IMG_DIR" --predictions_dir_path "$PRED_DIR" --output_dir_path "$PRED_DIR/level-1-raw"

run_in_env grand_env_utils
    python utils/prepare_level_1.py --image_dir_path "$IMG_DIR" --raw_dir_path "$PRED_DIR/level-1-raw" --output_dir_path "$PRED_DIR/level-1-processed"


# -------------------------------------------------------------------------------------------------------------------- #

# 11. Captioning using BLIP-2
run_in_env_targeted grand_env_3 pushd level_2_inference/1_blip-2
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR"
popd

# 12. Captioning using LLaVA
run_in_env grand_env_6 pushd level_2_inference/2_llava
    python infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --gpu_ids "$GPU_IDs" --llava_model_path "$CKPT_DIR/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
popd

# 13. Grounding using MDETR
run_in_env_targeted grand_env_7 pushd level_2_inference/3_mdetr
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --output_dir_path "$PRED_DIR" --blip2_pred_path "$PRED_DIR/blip2"  --llava_pred_path "$PRED_DIR/llava"
popd

# 14. Generate Level-2 Scene Graph and Update Level-1
run_in_env grand_env_utils
    python utils/merge_json_level_2.py --predictions_dir_path "$PRED_DIR" --output_dir_path "$PRED_DIR/level-2-raw"

run_in_env grand_env_utils
    python utils/prepare_level_2.py --raw_dir_path "$PRED_DIR/level-2-raw" --level_2_output_dir_path "$PRED_DIR/level-2-processed" --level_1_dir_path "$PRED_DIR/level-1-processed"


# -------------------------------------------------------------------------------------------------------------------- #

# 15. Enrich Attributes using GPT4RoI
run_in_env grand_env_8 pushd level_2_inference/4_gpt4roi/GPT4RoI
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env gpt4roi/infer.py --image_dir_path "$IMG_DIR" --level_2_pred_path "$PRED_DIR/level-2-processed" --output_dir_path "$PRED_DIR/level-2-processed_gpt4roi"
popd

# 16. Label Assignment using EVA-CLIP
run_in_env_targeted grand_env_4 pushd level_2_inference/5_label_assignment
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env infer.py --image_dir_path "$IMG_DIR" --level_2_dir_path "$PRED_DIR/level-2-processed_gpt4roi" --output_dir_path "$PRED_DIR/level-2-processed_eva_clip"
popd

# 17. Merge EVA-CLIP Assigned Labels & Calculate and Store Depths for All Objects
run_in_env_targeted grand_env_utils
    python utils/merge_eva_labels.py --level_2_dir_path "$PRED_DIR/level-2-processed_gpt4roi"  --labels_path "$PRED_DIR/level-2-processed_eva_clip" --output_dir_path "$PRED_DIR/level-2-processed_labelled" --store_depth --depth_map_dir "$PRED_DIR/midas"


# -------------------------------------------------------------------------------------------------------------------- #

# 18. Generate Level-3 Dense Captions
run_in_env grand_env_9 pushd level_3_dense_caption
    python run.py --image_dir_path "$IMG_DIR" --level_2_dir_path "$PRED_DIR/level-2-processed_labelled" --output_dir_path "$PRED_DIR/level-3-vicuna-13B" --gpu_ids "$GPU_IDs" --job_id '111'
popd

# 19. Generate Level-4 Additional Context
run_in_env grand_env_9 pushd level_4_extra_context
    python run.py --image_dir_path "$IMG_DIR" --level_2_dir_path "$PRED_DIR/level-2-processed_labelled" --output_dir_path "$PRED_DIR/level-4-vicuna-13B" --gpu_ids "$GPU_IDs" --job_id '111'
popd


# -------------------------------------------------------------------------------------------------------------------- #

# 20. Ground short & dense captions
run_in_env_targeted grand_env_utils
    python utils/ground_short_captions.py --data_dir_path "$PRED_DIR/level-2-processed_labelled" --output_dir_path "$PRED_DIR/short_captions_grounded"

run_in_env_targeted grand_env_utils
    python utils/ground_dense_caption.py --level_3_dense_caption_txt_dir_path "$PRED_DIR/level-3-vicuna_13B" --level_2_processed_json_path "$PRED_DIR/short_captions_grounded" --output_dir_path "$PRED_DIR/dense_captions_grounded"

# 21. Add Masks to the Annotations (sources: SAM Annotations & EVA Detector)
run_in_env_targeted grand_env_utils
    python utils/add_masks_to_annotations.py --input_dir_path "$PRED_DIR/dense_captions_grounded" --sam_json_dir_path "$SAM_ANNOTATIONS_DIR" --eva_02_pred_dir_path "$PRED_DIR/eva-02-01" --output_dir_path "$PRED_DIR/level-3-processed"

# 22. Use HQ-SAM for the Rest of the Masks not Found in SAM Annotations or EVA Detections
run_in_env_targeted grand_env_1 pushd utils/hq_sam
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUs" --master_port="$MASTER_PORT" --use_env run.py --image_dir_path "$IMG_DIR" --level_3_processed_path "$PRED_DIR/level-3-processed" --output_dir_path "$PRED_DIR/level-3-processed_with_masks" --checkpoints_path "$CKPT_DIR/sam_hq_vit_h.pth"
popd

# 23. Add Additional Context to the Annotations
run_in_env_targeted grand_env_utils
    python utils/add_addional_context.py --annotations_dir_path "$PRED_DIR/level-3-processed_with_masks" --level_4_additional_context_path "$PRED_DIR/level-4-vicuna_13B" --output_dir_path "$PRED_DIR/level-4-processed"


# -------------------------------------------------------------------------------------------------------------------- #

echo The pipeline inference completed and the predictions are saved in "$PRED_DIR/level-4-processed"
