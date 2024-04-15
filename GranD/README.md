# GranD - Grounding Anything Dataset 🚀
For details on downloading the dataset, preprocessing annotations for pre-training, and the automated annotation pipeline, please refer to [GranD.md](../docs/GranD.md) in the documentation.

## Running the GranD Automated Annotation Pipeline
The GranD automated annotation pipeline comprises four levels and a total of 23 steps. Each level utilizes multiple state-of-the-art (SoTA) vision-language models and pipeline scripts to construct image-scene graphs from raw predictions.

For a step-by-step guide on running the pipeline, refer to [run_pipeline.sh](run_pipeline.sh). The environments required are listed under [environments](environments).

### Create All Environments
There are ten environment `.yml` files provided in the [environments](environments) directory. Create all ten environments using the following commands:

```bash
conda env create -f grand_env_1.yml
conda env create -f grand_env_2.yml
...
...
conda env create -f grand_env_9.yml
conda env create -f grand_env_utils.yml
```

**NOTE:** While creating any of the above environments, if one or more `pip` dependencies fail to install, you may need to remove those dependencies from the environment file and rerun the command.

### Download Model Checkpoints
Download all required model checkpoints to your `CKPT_DIR` directory:

```bash
# For Landmark Detection
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3

# For Depth Estimation
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt

# For Image Tagging
Download from [recognize-anything/tag2text_swin_14m.pth](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/tag2text_swin_14m.pth) & [recognize-anything/ram_swin_large_14m.pth](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)

# For Co-DETR Detector
Download using this [Google Drive link](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) to obtain the `co_deformable_detr_swin_large_900q_3x_coco.pth` checkpoints.

# For EVA-02 Detector
Download from [eva02_L_lvis_sys.pth](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys.pth) & [eva02_L_lvis_sys_o365.pth](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys_o365.pth)

# For POMP
Download from [Google Drive](https://drive.google.com/file/d/1C8oU6cWkJdU3Q3IHaqTcbIToRLo9bMnu/view?usp=sharing) & [Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth](https://drive.google.com/file/d/1TwrjcUYimkI_f9z9UZXCmLztdgv31Peu/view?usp=sharing)

# For GRIT
wget -c https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth

# For OV-SAM
Download from [HarborYuan/ovsam_models/blob/main/sam2clip_vith_rn50x16.pth](https://huggingface.co/HarborYuan/ovsam_models/blob/main/sam2clip_vith_rn50x16.pth)

# For GPT4RoI
Follow the instructions at [GPT4RoI/Weights](https://github.com/jshilong/GPT4RoI?tab=readme-ov-file#weights) to obtain the GPT4RoI weights.
```

### Automatically Annotate Images
Refer to the [run_pipeline.sh](run_pipeline.sh) script for details. Below is a sample command to run the pipeline:

```bash
bash run_pipeline.sh $IMG_DIR $PRED_DIR $CKPT_DIR $SAM_ANNOTATIONS_DIR
```

Where:

1. `IMG_DIR` is the path to the directory containing images you wish to annotate.
2. `PRED_DIR` is the path to the directory where the predictions will be saved.
3. `CKPT_DIR` is the path to the directory containing all the checkpoints. For downloading the checkpoints, consult the README of each respective model.
4. `SAM_ANNOTATIONS_DIR` is the path to the directory containing SAM annotations (.json file).

**Note:** If you are not annotating SAM images, remove `ov-sam` from the pipeline and adjust the `add_masks_to_annotations.py` script accordingly. In this case, `SAM_ANNOTATIONS_DIR` will not be required.

### Disclaimer: 
We acknowledge that the pipeline is complex due to the involvement of many different models with various dependencies. Contributions that simplify or improve the pipeline are welcome.