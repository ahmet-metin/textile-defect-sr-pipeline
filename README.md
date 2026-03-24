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
