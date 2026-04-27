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
- **Input Resolution:** 128x128 pixels

## Performance Metrics
The model achieved high-precision segmentation results on the unseen validation set:
- **Mean Intersection over Union (IoU):** 0.9061
- **Dice Coefficient:** 0.9506

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone [https://github.com/Hiiianny/UNet-Human-Movement-Segmentation.git](https://github.com/Hiiianny/UNet-Human-Movement-Segmentation.git)
cd UNet-Human-Movement-Segmentation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit Application:
```bash
streamlit run app.py
```

## 🧠 Challenges & Learnings
During the development of this baseline, standard BCE loss resulted in blurry edges. Implementing a Hybrid Loss (BCE + Dice) significantly improved boundary sharpness. Furthermore, testing with out-of-distribution images (e.g., half-body shots or complex clothing) revealed a scale bias, emphasizing the necessity of diverse datasets and data augmentation for real-world robustness.
