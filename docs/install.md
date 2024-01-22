# Installation 🛠️
We recommend setting up a conda environment for the project:

```bash
conda create --name=glamm python=3.10
conda activate glamm

git clone https://github.com/mbzuai-oryx/groundingLMM.git
cd groundingLMM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# Install mmcv
git clone https://github.com/open-mmlab/mmcv
cd mmcv
git checkout v1.4.7
MMCV_WITH_OPS=1 pip install -e .

export PYTHONPATH="./:$PYTHONPATH"
```

In addition, we also provide conda environment contents in a `.zip` file. Please follow the below steps to set up the environment,

1. Download `glamm_conda_env.zip` from the [google_drive link](https://drive.google.com/file/d/1BN10oChcoKDDd0zC8tU88JcrfmLpKpkB/view?usp=sharing).
2. Extract the downloaded `zip` file: 
```bash
unzip glamm_conda_env.zip
```
3. Activate the environment: 
```bash
conda activate glamm
```


