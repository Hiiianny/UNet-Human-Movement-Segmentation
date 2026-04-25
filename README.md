# U-Net Human Movement Segmentation

This repository contains a PyTorch implementation of the **U-Net architecture** for full-body human segmentation. Built as a technical baseline for human movement analysis, this model is designed to isolate articulated human subjects in dynamic motion environments. 

An interactive web application built with **Streamlit** is also included for real-time inference demonstration.

## Project Overview
The primary goal of this project is to accurately segment human figures from complex backgrounds during extreme or dynamic movements. By utilizing the **TikTok Dances Segmentation Dataset**, the model learns to recognize diverse poses, overcoming the spatial bias typically found in standard static portrait datasets.

## Technical Specifications
- **Architecture:** U-Net (Encoder-Decoder with Skip Connections)
- **Framework:** PyTorch & Streamlit
- **Dataset:** TikTok Dances Segmentation (2,615 images)
- **Loss Function:** Hybrid BCE + Dice Loss (for sharp boundary delineation)
- **Optimization:** Adam Optimizer with `ReduceLROnPlateau` scheduler
- **Training Strategy:** Early Stopping implemented (converged at 30 epochs)
- **Input Resolution:** 128x128 pixels

## Performance Metrics
The model achieved high-precision segmentation results on the unseen validation set:
- **Mean Intersection over Union (IoU):** 0.9061
- **Dice Coefficient:** 0.9506

## 💻 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/Hiiianny/UNet-Human-Movement-Segmentation.git](https://github.com/Hiiianny/UNet-Human-Movement-Segmentation.git)
cd UNet-Human-Movement-Segmentation
