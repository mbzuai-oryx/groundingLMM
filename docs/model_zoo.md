# GLaMM Model Zoo 🚀

Welcome to the GLaMM Model Zoo! This repository contains a collection of state-of-the-art models from the GLaMM (Pixel Grounding Large Multimodal Model) family. Each model is designed for specific tasks in the realm of multimodal learning, combining visual and textual data processing.

## Models Overview

The following table provides an overview of the available models in our zoo. For each model, you can find links to its Hugging Face page. 

- To evaluate the pretrained models, please follow the instructions at [evaluation.md](evaluation.md).
- To run offline demo, please follow the instructions at [offline_demo.md](offline_demo.md).

| Model Name           | Hugging Face Link                                                                                         | Summary                                                                                                  |
|----------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| GLaMM-GranD-Pretrained | [Hugging Face](https://huggingface.co/MBZUAI/GLaMM-GranD-Pretrained) | Pretrained on GranD dataset.                                                          |
| GLaMM-FullScope      | [Hugging Face](https://huggingface.co/MBZUAI/GLaMM-FullScope)              | Model recommended for offline demo.                                                  |
| GLaMM-GCG            | [Hugging Face](https://huggingface.co/MBZUAI/GLaMM-GCG)                     | Finetuned on GranD-f dataset for GCG task.                                            |
| GLaMM-RefSeg         | [Hugging Face](https://huggingface.co/MBZUAI/GLaMM-RefSeg)                  | Finetuned on RefCOCO, RefCOCO+ and RefCOCOg datasets for referring expression segmentation task. |
| GLaMM-RegCap-RefCOCOg | [Hugging Face](https://huggingface.co/MBZUAI/GLaMM-RegCap-RefCOCOg) | Finetuned on RefCOCOg for region captioning task.                                    |
| GLaMM-RegCap-VG      | [Hugging Face](https://huggingface.co/MBZUAI/GLaMM-RegCap-VG)               | Finetuned on Visual Genome dataset for region captioning task.                       | 

Note that all models are finetuned on `GLaMM-GranD-Pretrained`.