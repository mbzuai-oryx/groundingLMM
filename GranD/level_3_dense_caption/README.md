## Installation

Please follow the instructions at the below links to install FastChat and VLLM
1. FastChat: [https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)
2. VLLM: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)


## Command to Run

```bash
python run.py --image_dir_path <path to the directory containing images> --level_2_dir_path <path to level-2 processed files> --output_dir_path <base path to store the predictions> --gpu_ids <comma separated gpu ids to run inference on multiple gpus> --job_id '111'

```
For help, run
```bash
python run.py -h
python query_vicuna_vLLM_level_3.py -h
```
