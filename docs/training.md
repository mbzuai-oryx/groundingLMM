# Training GLaMM 🚀
GLaMM is pre-trained on the GranD dataset and then fine-tuned on multiple downstream tasks including Grounded Conversation Generation (GCG), referring expression segmentation, region-level captioning, and image-level captioning using OpenSource datasets.

## Downstream Task-Specific Training 🛠️

This section explains how to perform downstream fine-tuning using the pretrained GLaMM model checkpoints.

### Preparing the OpenSource Datasets 📂

Refer to the [datasets readme](../docs/datasets.md) for details on organizing the data.

Generic settings:
- Path to the GLaMM GranD pretrained Hugging Face model: `PRETRAINED_HF_PATH=MBZUAI/GLaMM-GranD-Pretrained`
- Path to the Grounding Image Encoder Checkpoints (SAM pretrained weights): `GROUNDING_ENC_CKPT_PATH=./checkpoints/sam_vit_h_4b8939.pth`

### 1) Grounded Conversation Generation (GCG) 🗨️

For GCG, the model is fine-tuned on two types of datasets: (i) GranD-f Dataset and (ii) Semantic Segmentation Datasets.
  - [GranD-f datasets](../docs/datasets.md#1-grand-f-grounded-conversation-generation-gcg-dataset): RefCoco_GCG, PSG_GCG, Flickr_GCG, GranDf_GCG
  - [Semantic Segmentation Datasets](../docs/datasets.md#2-semantic-segmentation-datasets): ade20k, cocostuff, pascal_part, paco_lvis, mapillary

```bash
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Semantic_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG" --segm_sample_rates "1,3,3,3,1" --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" --epochs 10 --steps_per_epoch 500 --mask_validation
```

### 2) Region-level Captioning 🖼️ 

For region-level captioning, the model is fine-tuned on specific datasets:
  - [Region-level Captioning Dataset](../docs/datasets.md#4-region-level-captioning-datasets-expression-generation): RefCocoG_Reg, VisGenomeRegVal

For RefCOCOg:
```bash
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_reg_data --reg_dataset 'RefCocoG_Reg' --reg_sample_rates "1" --val_dataset 'RefCOCOgRegVal' --epochs 5 --steps_per_epoch 500
```
For Visual Genome:
```bash
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_reg_data --reg_dataset 'VisGen_Reg' --reg_sample_rates "1" --val_dataset 'VisGenomeRegVal' --epochs 5 --steps_per_epoch 500
```

### 3) Referring Expression Segmentation 🎯

For results on RefCOCO, RefCOCO+ and RefCOCOg datasets, the model is fine-tuned using the following datasets:
  - [Referring Expression Dataset](../docs/datasets.md#3-referring-expression-datasets): refcoco, refcoco+, refcocog

```bash
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Refer_Segm" --segm_sample_rates "1" --refer_segm_data "refcoco||refcoco+||refcocog" --val_dataset "RefCOCOgSegVal" --epochs 5 --steps_per_epoch 350 --mask_validation
```

### 4) Finetuning on Combined Tasks 🌍
To enable combined capabilities in tasks like Grounded Conversation Generation (GCG), Image-level captioning, Visual-question answering, Region-level captioning, and Referring Expression Segmentation, finetune GLaMM using a mix of open-source datasets. This training replicates the model used in the demo.

Refer to [datasets readme](../docs/datasets.md) for data preparation details.

The `train.py` script is pre-configured with default argument values optimized for combined open-source training. However, for clarity and customization, we detail all essential arguments below:

```bash
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_cap_data --use_reg_data --use_segm_data -cap_dataset "CocoCap||LLaVaInstruct" --cap_sample_rate "1,2" --reg_dataset "RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg||FlickrGCGVal" --reg_sample_rates -"1,1,1,1,1" -seg_dataset "Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG" --segm_sample_rates "4,3,2,2,2,1" --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" --epochs 10 --steps_per_epoch 500
```

### Merge LORA Weights
We use LORA finetuning for downstream tasks. Please follow the instructions below to merge LORA weights after training.

After training the saved checkpoints directory will look like,
```
├── global_step5000
│   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_5_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt
│   ├── bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt
```
Run the following command to merge LORA weights,

```bash
python zero_to_fp32.py . ./pytorch_model.bin

# From the root directory
export PYTHONPATH="./:$PYTHONPATH"
python scripts/merge_lora_weights.py --version 'MBZUAI/GLaMM-GranD-Pretrained' --weight 'path/to/pytorch_model.bin' --save_path 'path/to/save/the/merged/model/in/HF/format'  
```