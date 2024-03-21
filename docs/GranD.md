# GranD - Grounding Anything Dataset 🚀
The [Grounding-anything](https://grounding-anything.com/) Dataset (GranD) dataset offers densely annotated data, acquired through an automated annotation pipeline that leverages state-of-the-art (SOTA) vision and V-L models. This documentation covers how to download the GranD dataset and a guide to the automated annotation pipeline used to create GranD.  

## Download GranD 📂
- Annotations: [MBZUAI/GranD](https://huggingface.co/datasets/MBZUAI/GranD)
- Images: [Download](https://ai.meta.com/datasets/segment-anything-downloads/)
GranD utilizes images from the SAM dataset. 

Note: Please note that annotations are being uploaded incrementally and more parts will be available soon.

### Preparing the Pretraining Annotations from GranD 🛠️

After downloading the GranD annotations, utilize the scripts below to transform them into GLaMM pretraining data, or to prepare them for your specific tasks.

- For object-level tasks like object detection, semantic segmentation: [prepare_object_lvl_data.py](../GranD/prepare_annotations/prepare_object_lvl_data.py)
- For image-level captioning and caption grounding: [prepare_grand_caption_grounding.py](../GranD/prepare_annotations/prepare_grand_caption_grounding.py)
- For referring expression generation and referring expression segmentation: [prepare_grand_referring_expression](../GranD/prepare_annotations/prepare_grand_referring_expression.py)

The above scripts generate annotations in JSON format. To convert these for use in pretraining datasets requiring LMDB format, please use to the following scripts:
- To convert to lmdb: [get_txt_for_lmdb.py](../GranD/prepare_annotations/get_txt_for_lmdb.py)
- To extract file names in txt format: [get_txt_for_lmdb.py](../GranD/prepare_annotations/get_txt_for_lmdb.py)

### GranD Automated Annotation Pipeline

GranD is a comprehensive, multi-purpose image-text dataset offering a range of contextual information, from fine-grained to high-level details. The pipeline contains four distinct levels.
The code for the four levels are provided in: [GranD](../GranD)

More detailed information:
- To run the entire pipeline: [run_pipeline.sh](../GranD/run_pipeline.sh)
- To set up the environments detailed in [run_pipeline.sh](../GranD/run_pipeline.sh) refer to : [environments](../GranD/environments)
- Level-1 : Object Localization and Attributes
  - Landmark Categorization: [landmark](../GranD/level_1_inference/1_landmark_categorization/README.md)
  - Depth Map Estimation: [Midas Depth Estimation](../GranD/level_1_inference/2_depth_maps/README.md)
  - Image Tagging: [RAM Tag2Text Tagging](../GranD/level_1_inference/3_image_tagging/README.md)
  - Standard Object Detection: [CO-DETR OD](../GranD/level_1_inference/4_co_detr/README.md), [EVA OD](../GranD/level_1_inference/4_co_detr/README.md)
  - Open Vocabulary Object Detection: [OWL-ViT OVD](../GranD/level_1_inference/6_owl_vit), [POMP OVD](../GranD/level_1_inference/7_pomp)
  - Attribute Detection and Grounding: [Attribute & Grounidng GRiT](../GranD/level_1_inference/8_grit/README.md)
  - Open Vocabulary Classification: [OV Classification OV-SAM](../GranD/level_1_inference/9_ov_sam/README.md)
  - Combine the predictions: [Merging](../GranD/utils/merge_json_level_1_with_nms.py)
  - Generate Level-1 Scene Graph: [Level-1 Scene Graph](../GranD/utils/prepare_level_1.py)
- Level-2: Relationships
  - Captioning: [BLIP-2 Captioning](../GranD/level_2_inference/1_blip-2/README.md), [LLaVA Captioning](../GranD/level_2_inference/2_llava/README.md)
  - Grounding Short Captions: [MDETR Grounding](../GranD/level_2_inference/3_mdetr/README.md)
  - Combine the predictions: [Merging](../GranD/utils/merge_json_level_2.py)
  - Generate Level-2 Scene Graph and Update Level-1: [Level-2 Scene Graph](../GranD/utils/prepare_level_2.py)
  - Enrich Attributes: [GPT4-RoI Attributes](../GranD/level_2_inference/4_gpt4roi/README.md)
  - Label Assignment: [EVA-CLIP Label Assignment](../GranD/level_2_inference/5_label_assignment/README.md)
- Level-3: Scene Graph and Dense Captioning
  - Generate Dense Captions: [Scene graph dense captioning LLaVA](../GranD/level_3_dense_caption/README.md)
- Level-4: Extra Contextual Insight: 
  - Generate Level-4 Additional Context: [Extra Context](../GranD/level_4_extra_context/README.md)


