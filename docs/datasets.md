# Prepare Dataset 🚀
This guide outlines the datasets required for opensource fine-tuning of GLaMM, which encompasses tasks like Grounded Conversation Generation (GCG), Image-level captioning, Visual-question answering, Region-level captioning, and Referring Expression Segmentation. These datasets are used for fine-tuning to achieve the model demonstrated in our demo. We will also highlight the specific datasets needed for each task.

To achieve all the capabilities of GLaMM, the following dataset types are used:
1. GranD-f Grounded Conversation Generation (GCG) Dataset
2. Semantic Segmentation Datasets
3. Referring Expression Datasets (Expression Comprehension)
4. Region-level Captioning Datasets (Expression Generation)
5. Image Captioning
6. Visual Question Answering
7. GranD pretraining Datasets

Overall, they must be arranged in the following format:
```
├── GranDf
│   ├── annotations
│   │   ├── train
│   │   │   ├── GranDf_HA_GCG_train.json
│   │   │   ├── OpenPsgGCG_train.json
│   │   │   ├── OpenPsgGCG_val.json
│   │   │   ├── RefCOCOg_GCG_train.json
│   │   │   ├── RefCOCOg_GCG_val.json
│   │   │   ├── flickr_mergedGT_GCG_train.json
│   │   │   ├── flickr_mergedGT_GCG_val.json
│   │   ├── val_test
│   │   │   ├── test_gcg_coco_caption_gt.json
│   │   │   ├── test_gcg_coco_mask_gt.json
│   │   │   ├── val_gcg_coco_caption_gt.json
│   │   │   ├── val_gcg_coco_mask_gt.json
├── GranDf_HA_images
│   ├── train
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
│   ├── val_test
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
│
├── Semantic_Segm
│   ├── ade20k
│   │   ├── annotations
│   │   │   ├── training
│   │   │   │   ├── ADE_train_00000001.png
│   │   │   │   ├── ADE_train_00000002.png
│   │   ├── images
│   │   │   ├── training
│   │   │   │   ├── ADE_train_00000001.jpg
│   │   │   │   ├── ADE_train_00000002.jpg
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── training
│   │   │   ├── v2.0
│   │   │   │   ├── labels
│   │   │   │   │   ├── 0035fkbjWljhaftpVM37-g.png
│   │   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.png
│   │   │   ├── images
│   │   │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
│   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
├── paco_lvis
│   │   ├── annotations
│   │   │   ├── paco_lvis_v1_train.json
├── pascal_part
│   │   ├── train.json
│   │   ├── VOCdevkit
│   │   │   │   ├── VOC2010
│   │   │   │   │   ├── JPEGImages
│   │   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   │   ├── 2007_000032.jpg
│
├── Refer_Segm
│   ├── refcoco
│   ├── refcoco+
│   ├── refcocog
│   ├── refclef
│   ├── images
│   │   ├── saiapr_tc-12
│   │   │   ├── 00
│   │   │   ├── 01
│
├── RefCoco_Reg
│   ├── mdetr_annotations
│   │   ├── finetune_refcoco_train.json
│   │   ├── finetune_refcocog_train.json
│   │   ├── finetune_refcocog_val.json
│   │   ├── finetune_refcoco+_train.json
│   │   ├── final_flickr_mergedGT_train.json
├── visual_genome
│   │   ├── test_caption.json
│   │   ├── train.json
│   │   ├── images
│   │   │   ├── 1000.jpg
│   │   │   ├── 1001.jpg
│
├── llava_dataset
│   ├── llava_instruct_150k.json
│
├── coco_2017
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   ├── annotations
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
│
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
│
├── flikcr_30k
│   ├── train
│   │   ├── 1000092795.jpg
│   │   ├── 10002456.jpg
```

### 1) GranD-f Grounded Conversation Generation (GCG) Dataset
The [GranD-f](https://grounding-anything.com/GranD-f) datasets comprise four datasets: one high-quality human-annotated set proposed in our GLaMM paper, and 3 other datasets repurposed for the GCG task.

Download links and structure:
- Annotations: [MBZUAI/GranD-f](https://huggingface.co/datasets/MBZUAI/GranD-f)
- Images: `GranDf_HA_images` [Download](https://drive.google.com/file/d/1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_/view?usp=drive_link)
- Other necessary datasets: 
  - Open-PSG GCG: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))
  - RefCOCO-g GCG: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
  - Flickr-30k GCG: `flikcr_30k` - flikcr_30k (train) - Download the train images from the [Flickr30K webpage](https://shannon.cs.illinois.edu/DenotationGraph/) or use download from the following [link](https://drive.google.com/file/d/1iomUn-Ht0OBfieMuyoVqEFj5PEmXfQ0U/view?usp=drive_link).

```
├── GranDf
│   ├── annotations
│   │   ├── train
│   │   │   ├── GranDf_HA_GCG_train.json
│   │   │   ├── OpenPsgGCG_train.json
│   │   │   ├── OpenPsgGCG_val.json
│   │   │   ├── RefCOCOg_GCG_train.json
│   │   │   ├── RefCOCOg_GCG_val.json
│   │   │   ├── flickr_mergedGT_GCG_train.json
│   │   │   ├── flickr_mergedGT_GCG_val.json
│   │   ├── val_test
│   │   │   ├── test_gcg_coco_caption_gt.json
│   │   │   ├── test_gcg_coco_mask_gt.json
│   │   │   ├── val_gcg_coco_caption_gt.json
│   │   │   ├── val_gcg_coco_mask_gt.json
├── GranDf_HA_images
│   ├── train
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
│   ├── val_test
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
├── coco_2017
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
├── flikcr_30k
│   ├── train
│   │   ├── 1000092795.jpg
│   │   ├── 10002456.jpg
```

### 2) Semantic Segmentation Datasets
For semantic segmentation, we use five open-source datasets providing segmentation masks and semantic class labels: - ADE20K, COCO-Stuff, PASCAL-Part, PACO-LVIS, and Mapillary. 

Download links and structure:
- [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
- [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
- [PASCAL-Part](https://www.mapillary.com/dataset/vistas)
- [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup)
- [Mapillary](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part)
- COCO images: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))

Download and arrange as shown in the directory structure below.

```
├── Semantic_Segm
│   ├── ade20k
│   │   ├── annotations
│   │   │   ├── training
│   │   │   │   ├── ADE_train_00000001.png
│   │   │   │   ├── ADE_train_00000002.png
│   │   ├── images
│   │   │   ├── training
│   │   │   │   ├── ADE_train_00000001.jpg
│   │   │   │   ├── ADE_train_00000002.jpg
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── training
│   │   │   ├── v2.0
│   │   │   │   ├── labels
│   │   │   │   │   ├── 0035fkbjWljhaftpVM37-g.png
│   │   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.png
│   │   │   ├── images
│   │   │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
│   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
├── paco_lvis
│   │   ├── annotations
│   │   │   ├── paco_lvis_v1_train.json
├── pascal_part
│   │   ├── train.json
│   │   ├── VOCdevkit
│   │   │   │   ├── VOC2010
│   │   │   │   │   ├── JPEGImages
│   │   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   │   ├── 2007_000032.jpg
├── coco_2017
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
```

### 3) Referring Expression Datasets
For Referring Expression segmentation - we use COCO referring expression comprehension datasets: RefCOCO, RefCOCO+, RefCOCOg, and RefCLEF.

Download links and structure:
- [RefCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
- [RefCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
- [RefCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)
- [RefCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip)
- RefCOCO images: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
- For RefCLEF, you need images `[saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)`

Download the data from the source links, and arrange as follows:

```
├── Refer_Segm
│   ├── refcoco
│   ├── refcoco+
│   ├── refcocog
│   ├── refclef
│   ├── images
│   │   ├── saiapr_tc-12
│   │   │   ├── 00
│   │   │   ├── 01
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
```

### 4) Region-level Captioning Datasets (Expression Generation)
For region-level captioning, we use five open source datasets with region(bbox) grounding: RefCOCO, RefCOCOg, RefCOCO+, Visual Genome(V1.2) and Flickr30K. 

Download links and structure:
- Annotations - mdetr_annotations: [Download](https://drive.google.com/file/d/1gvH5ToNtmIr3qz7C9lNi_fDmElwAANsI/view?usp=drive_link)
- Visual Genome: [train.json](https://datarelease.blob.core.windows.net/grit/VG_preprocessed_annotations/train.json), [test_caption.json](https://drive.google.com/file/d/1zF3UGHU1rvgTujinqJ-hZtrCBVsfsuel/view?usp=sharing) [images](https://nlp.stanford.edu/data/gqa/images.zip)
- Flickr30k: Download the train images from the [Flickr30K webpage](https://shannon.cs.illinois.edu/DenotationGraph/) or use download from the following [link](https://drive.google.com/file/d/1iomUn-Ht0OBfieMuyoVqEFj5PEmXfQ0U/view?usp=drive_link).
- RefCOCO images: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
Download the data from the source links, and arrange as follows:

```
├── RefCoco_Reg
│   ├── mdetr_annotations
│   │   ├── finetune_refcoco_train.json
│   │   ├── finetune_refcocog_train.json
│   │   ├── finetune_refcocog_val.json
│   │   ├── finetune_refcoco+_train.json
│   │   ├── final_flickr_mergedGT_train.json
├── visual_genome
│   │   ├── test_caption.json
│   │   ├── train.json
│   │   ├── images
│   │   │   ├── 1000.jpg
│   │   │   ├── 1001.jpg
├── flikcr_30k
│   ├── train
│   │   ├── 1000092795.jpg
│   │   ├── 10002456.jpg
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
```

### 5) Image Captioning
We use the COCO caption dataset.

Download links and structure:
- Annotations - [COCO - 2017 annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- Images: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))

Structure as shown in the directory structure above.

```
├── coco_2017
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   ├── annotations
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
```

### 6) Visual Question Answering
We use the LLaVA-instruct-150k set for visual question answering. Download and arrange as detailed below.

Download links and structure:
- Annotations - [LLaVA-instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)
- Images: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))

```
├── llava_dataset
│   ├── llava_instruct_150k.json
├── coco_2017
│   ├── train2017
```

### 7) GranD pretraining Datasets

We convert the GranD dataset to multiple annotations in LMDB form for pretraining based on the tasks. For details on how to prepare the annotations, please refer to: [Pretraining Annotations from GranD](../docs/GranD.md#preparing-the-pretraining-annotations-from-grand-).

- For image-level captioning:
  - Short Captioning: [GrandShortCaptionDataset](../dataset/caption_datasets/GranD_ShortCaption_ds.py)
- For referring expression generation and referring expression segmentation:
  - Region-level captioning (referring expression generation): [GrandReferRegDataset](../dataset/region_datasets/GranD_ReferringRegion_ds.py)
  - Referring expression segmentation: [GrandReferSegmDataset](../dataset/segm_datasets/GranD_ReferringSegm_ds.py)


