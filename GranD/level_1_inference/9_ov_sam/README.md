# Setup the Environment

Please follow the instructions at [Open Vocabulary SAM Installation](https://github.com/HarborYuan/ovsam?tab=readme-ov-file#%EF%B8%8F-installation) for the installation.

# Download Pretrained Checkpoints

Download checkpoints from [HarborYuan/ovsam_models/blob/main/sam2clip_vith_rn50x16.pth](https://huggingface.co/HarborYuan/ovsam_models/blob/main/sam2clip_vith_rn50x16.pth) and place it in the current directory.

## Command to Run

```bash
python launch_ov_sam_multi_gpu_inference.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --sam_annotations_dir <path to the directory containing all sam annotations> --gpu_ids <comma separated gpu ids to run inference on multiple gpus>
```

For help, run

```bash
python infer.py -h
```
