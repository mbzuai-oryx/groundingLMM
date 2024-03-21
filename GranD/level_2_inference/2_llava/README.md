## Install LLaVA

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

git checkout v1.1.3
pip install -e .

```

## Download Pretrained Checkpoints

```bash
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
```

## Command to Run

```bash
python infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --gpu_ids <comma separated gpu ids to run inference on multiple gpus> --llava_model_path <path to the checkpoints - llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3>

```
For help, run
```bash
python infer.py -h
```