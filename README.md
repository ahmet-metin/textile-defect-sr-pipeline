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


