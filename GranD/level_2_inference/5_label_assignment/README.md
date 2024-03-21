## Installation

Please follow the instructions at [baaivision/EVA/tree/master/EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP#setup) for installation.


## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env infer.py --image_dir_path <path to the directory containing images> --level_2_dir_path <path to level-2 processed files> --output_dir_path <base path to store the predictions>

```
For help, run
```bash
python infer.py -h
```
