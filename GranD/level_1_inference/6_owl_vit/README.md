# Setup the Environment

Install transformers and pytorch.

# Download Pretrained Checkpoints

The checkpoints will be automatically be downloaded from the HuggingFace.

## Command to Run

```bash
python launch_owl_vit_multi_gpu_inference.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --tags_dir_path <path containing tag2text and ram predictions> --gpu_ids <comma separated gpu ids to run inference on multiple gpus>
```

For help, run

```bash
python infer.py -h
```
