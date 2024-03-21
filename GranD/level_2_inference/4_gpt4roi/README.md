## Installation

Please follow the steps below,
```bash
cd level_2_inference/4_gpt4roi
git clone https://github.com/jshilong/GPT4RoI.git

cp ddp.py GPT4RoI/gpt4roi
cp inference_utils.py GPT4RoI/gpt4roi
cp infer.py GPT4RoI/gpt4roi

cd GPT4RoI

```
Follow the instructions at [GPT4RoI/Install](https://github.com/jshilong/GPT4RoI?tab=readme-ov-file#install) for environment setup.

## Download Pretrained Checkpoints

Follow the instructions at [GPT4RoI/Weights](https://github.com/jshilong/GPT4RoI?tab=readme-ov-file#weights) to get GPT4RoI weights.

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env gpt4roi/infer.py --image_dir_path <path to the directory containing images> --level_2_pred_path <path to level-2 processed files> --output_dir_path <base path to store the predictions>

```
For help, run
```bash
python gpt4roi/infer.py -h
```