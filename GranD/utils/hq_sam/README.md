## Installation

Please follow the instructions at [SysCV/sam-hq](https://github.com/SysCV/sam-hq?tab=readme-ov-file#standard-installation) for installation.


## Download Pretrained Checkpoints
Please download the model checkpoints from [sam_hq_vit_h.pth](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing) and place in the current directory.

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env run.py --image_dir_path <path to the directory containing images> --level_3_processed_path <path to level-3 processed files> --output_dir_path <base path to store the predictions> --checkpoints_path <path to the sam_hq_vit_h.pth>

```
For help, run
```bash
python run.py -h
```
