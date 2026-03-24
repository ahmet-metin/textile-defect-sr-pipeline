# Real-Time Textile Defect Inspection: A Lightweight Super-Resolution Augmented Detection Pipeline

This repository contains the code and resources for a lightweight super-resolution-assisted pipeline for real-time textile defect detection and segmentation.

The repository is associated with the manuscript:

**Real-Time Textile Defect Inspection: A Lightweight Super-Resolution Augmented Detection Pipeline**

If you use this repository, please cite the related manuscript.

---

## Overview

Textile defect inspection in industrial production is often limited by low-resolution imaging, small defect visibility, and hardware cost. This repository provides a practical end-to-end pipeline that aims to improve inspection performance under low-cost imaging settings by combining:

- lightweight ESRGAN-based super-resolution,
- fixed-size patch/tile generation,
- YOLOv8-seg-based defect detection and segmentation,
- optional Raspberry Pi-oriented edge deployment.

The main idea is to enhance low-resolution textile images before downstream analysis, divide the enhanced images into local tiles, and then perform object detection and segmentation on these tiles.

---

## Repository Structure

```text
textile-defect-sr-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── esrgan_train.yaml
├── src/
│   ├── edge/
│   │   └── rpi_demo.py
│   ├── esrgan/
│   │   ├── dataset.py
│   │   ├── infer_esrgan.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   ├── train_esrgan.py
│   │   └── utils.py
│   ├── pipeline/
│   │   └── tile_images.py
│   └── yolo/
│       ├── custom_data_seg.yaml
│       ├── infer_yolo_seg.py
│       └── train_yolo_seg.py
```

## Main Components

### 1. ESRGAN-based Super-Resolution

The ESRGAN module is used to enhance low-resolution textile images before inspection.

**Relevant files:**
- `src/esrgan/model.py`
- `src/esrgan/losses.py`
- `src/esrgan/dataset.py`
- `src/esrgan/train_esrgan.py`
- `src/esrgan/infer_esrgan.py`

### 2. Patch-Based / Tile-Based Analysis

The enhanced images are divided into fixed-size non-overlapping local patches for downstream processing.

**Relevant file:**
- `src/pipeline/tile_images.py`

### 3. YOLOv8-seg Defect Detection and Segmentation

Tile-level defect detection and segmentation are performed using YOLOv8-seg.

**Relevant files:**
- `src/yolo/train_yolo_seg.py`
- `src/yolo/infer_yolo_seg.py`
- `src/yolo/custom_data_seg.yaml`

### 4. Edge / Raspberry Pi Demo

A lightweight proof-of-concept deployment script is also included for Raspberry Pi-oriented experiments.

**Relevant file:**
- `src/edge/rpi_demo.py`

This script is provided as a deployment demo and is not required to reproduce the main training pipeline.

## Installation

It is recommended to use a clean Python environment.

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### On Windows

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Experimental Environment

The main experimental environment used in the study includes:

- Python 3.10.18
- OpenCV 4.7.0.68
- PyTorch 2.6.0+cu124
- Torchvision 0.21.1
- Ultralytics 8.3.176
- CUDA 12.4

Depending on your setup, you may need to adapt the installed PyTorch build to your CUDA or CPU environment.

## Data

This repository contains the code for the proposed pipeline.  
The datasets used in this study come from both public and custom sources.

### Public dataset

The TILDA fabric dataset used in this work was obtained from its original public source:

- [TILDA Fabric Dataset on Roboflow Universe](https://universe.roboflow.com/irvin-andersen/tilda-fabric)

Due to third-party ownership and licensing considerations, the TILDA dataset is **not redistributed** in this repository. Please access it from the original source.

### Custom dataset

In addition to public data, this study also uses custom-generated textile defect images and annotations produced during the experimental workflow.

If shared publicly, the custom dataset and related annotations should be provided through a persistent repository such as **Zenodo**.

You can later replace the placeholder below with the actual link:

- Custom dataset and annotations: **[Zenodo link will be added here]**

### Dataset configuration

The YOLO segmentation dataset paths are defined in:

```text
src/yolo/custom_data_seg.yaml
```

Please update dataset paths according to your local environment if needed.

---

## Weights

This repository can be used together with pretrained model weights for easier reproducibility.

Recommended weights to share:

- ESRGAN generator checkpoint used in the main experiments
- optional lightweight ESRGAN generator checkpoint used for Raspberry Pi experiments
- YOLOv8-seg best weights

You can later replace the placeholders below with actual links:

- ESRGAN generator weights: **[link will be added here]**
- Raspberry Pi ESRGAN weights: **[link will be added here]**
- YOLOv8-seg weights: **[link will be added here]**

If file sizes are large, it is recommended to share weights through **Zenodo** or **GitHub Releases** instead of committing them directly into the repository.

---

## Training

### Train ESRGAN

Edit the training configuration file:

```text
configs/esrgan_train.yaml
```

Then run:

```bash
python src/esrgan/train_esrgan.py
```

### Train YOLOv8-seg

```bash
python src/yolo/train_yolo_seg.py
```

---

## Inference

### Run ESRGAN inference

```bash
python src/esrgan/infer_esrgan.py
```

### Run YOLOv8-seg inference

```bash
python src/yolo/infer_yolo_seg.py
```

---

## Tiling / Patch Generation

The proposed pipeline uses fixed-size non-overlapping local patches for patch-based analysis.

Example utility:

```bash
python src/pipeline/tile_images.py
```

Please update input/output paths in the script if needed.

---

## Raspberry Pi Demo

The repository also includes a lightweight proof-of-concept deployment script for Raspberry Pi-oriented experiments:

```text
src/edge/rpi_demo.py
```

This script is intended as an edge deployment demo and is **not required** to reproduce the main training pipeline.

---

## Reproducibility Notes

To improve reproducibility:

- keep dataset paths configurable,
- preserve the model checkpoints used in the experiments,
- document whether inference is performed with pretrained weights or retrained models,
- report hardware details when comparing runtime,
- use the same image size and preprocessing settings described in the manuscript.

---

## Manuscript Relation

This repository is directly related to the manuscript:

**Real-Time Textile Defect Inspection: A Lightweight Super-Resolution Augmented Detection Pipeline**

If you use this repository, please cite the related manuscript.

---

## Citation

You can update this section after the manuscript is accepted or a DOI / preprint link becomes available.

### Placeholder citation

```bibtex
@article{metin2025textile,
  title={Real-Time Textile Defect Inspection: A Lightweight Super-Resolution Augmented Detection Pipeline},
  author={Metin, Ahmet and Ozkan, Haydar},
  journal={Under review},
  year={2025}
}
```

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.
