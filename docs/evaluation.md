# Evaluating GLaMM 🔍
This guide provides instructions on evaluating the pretrained GLaMM models on the downstream tasks including Grounded Conversation Generation (GCG), referring expression segmentation and region-level captioning.


### 1) Grounded Conversation Generation (GCG) 🗨️
Run the following instruction to evaluate GLaMM model on the GCG task

```bash
bash eval/gcg/run_evaluation.sh 'path/to/the/HF/checkpoints/path' 'path/to/the/directory/to/save/the/evaluation/results'
```

<p align="center">
  <img src="../images/tables/GCG_Table.png" alt="GCG_Table">
</p>


To evaluate provided finetuned GCG model, run,

```bash
bash eval/gcg/run_evaluation.sh 'MBZUAI/GLaMM-GCG' './results_gcg_finetuned'
```
This will automatically download the `MBZUAI/GLaMM-GCG` from HuggingFace.


### 2) Referring Expression Segmentation 🎯
Run the following instruction to evaluate GLaMM model on the referring expression segmentation task

```bash
bash eval/referring_seg/run_evaluation.sh 'path/to/the/HF/checkpoints/path' 'path/to/the/directory/to/save/the/evaluation/results'
```

To evaluate provided finetuned RefSeg model, run,

```bash
bash eval/referring_seg/run_evaluation.sh 'MBZUAI/GLaMM-RefSeg' './results_refseg_finetuned'
```
This will automatically download the `MBZUAI/GLaMM-RefSeg` from HuggingFace.


<p align="center">
  <img src="../images/tables/ReferSeg_Table.png" alt="Table_RefSeg">
</p>


### 3) Region-level Captioning 🖼️ 
Run the following instruction to evaluate GLaMM model on the region-level captioning task

#### RefCOCOg
```bash
bash eval/region_captioning/run_evaluation_RefCOCOg.sh 'path/to/the/HF/checkpoints/path' 'path/to/the/directory/to/save/the/evaluation/results'
```

To evaluate provided finetuned RefCOCOg model, run,

```bash
bash eval/region_captioning/run_evaluation_RefCOCOg.sh 'MBZUAI/GLaMM-RegCap-RefCOCOg' './results_regcap_refcocog_finetuned'
```
This will automatically download the `MBZUAI/GLaMM-RegCap-RefCOCOg` from HuggingFace.


#### Visual Genome
```bash
bash eval/region_captioning/run_evaluation_VG.sh 'path/to/the/HF/checkpoints/path' 'path/to/the/directory/to/save/the/evaluation/results'
```

To evaluate provided finetuned VG model, run,

```bash
bash eval/region_captioning/run_evaluation_VG.sh 'MBZUAI/GLaMM-RegCap-VG' './results_regcap_vg_finetuned'
```
This will automatically download the `MBZUAI/GLaMM-RegCap-VG` from HuggingFace.

<p align="center">
  <img src="../images/tables/Region_Cap_Table.png" alt="Table_RegionCap">
</p>
