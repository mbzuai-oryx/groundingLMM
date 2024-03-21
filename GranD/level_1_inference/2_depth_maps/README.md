## Installation

```bash
conda env create -f environment.yml
conda activate midas-py310

```

## Download Pretrained Checkpoints

```bash
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --model_weights <path to the checkpoints - dpt_beit_large_512.pt>

```
For help, run
```bash
python infer.py -h
```