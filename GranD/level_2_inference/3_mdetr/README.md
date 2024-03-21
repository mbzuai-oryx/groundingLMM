## Installation

Please follow the instructions at [ashkamath/mdetr](https://github.com/ashkamath/mdetr?tab=readme-ov-file#usage) for installation.

## Download Pretrained Checkpoints

The checkpoints will be automatically be downloaded.

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --blip2_pred_path <predictions dir path of blip2 captions>  --llava_pred_path <predictions dir path of llava captions>

```
For help, run
```bash
python infer.py -h
```