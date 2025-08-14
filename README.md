
# Where's Waldo? – Automated Visual Search using YOLOv11

![Find Waldo Illustration](/assets/images/1.jpg "Find Waldo Illustration")

## 📜 Overview

This project implements an automated visual search system to detect Waldo in dense illustrations using the YOLOv11 object detection model. Each illustration is split into 640×640 sub-images, with bounding box annotations for Waldo when present. The pipeline is optimized for small-object detection and rare-class scenarios, using tailored hyperparameters and targeted data augmentation.

---

## 📂 Project Structure

```bash
where-is-waldo/
├── assets/ # assets for readme
│ └── images/
│   └── 1.jpg
│
├── datasets/ # train and val datasets (populated from preprocess.py)
│ ├── train/ # 70% of dataset
│ │  ├── images/ # jpg files
│ │  └── labels/ # txt files
│ └── val/  # 20 % of dataset
│    ├── images/ # jpg files
│    └── labels/ # txt files
│
├── labelled_data/ # raw dataset from kaggle
│ ├── images/ # high-res original images
│ └── labels/ # resp labels of each img from Roboflow
│
├── Models/ # save trained YOLO models from train.py
│
├── tests/ # 10% testing dataset
│    ├── input/ # input data for test
│    └── output/ # output of test
│
├── .gitignore
├── customLib.py # python utility functions
├── DataAugmentation # generate training data****
├── inference.py # predict on test data
├── Main # generate training data****
├── Model # generate training data****
├── preprocess.py # preprocess data****
├── README.md 
├── requirements.txt
└── train.py # python script to train data
```

---

## 🚀 Features

1. Tiling of large illustrations into YOLO-compatible sub-images

2. Annotation handling for images using [RoboFLow](https://app.roboflow.com/)

3. YOLOv11n small-object optimization

4. Steps: preprocessing → training → evaluation → inference

---

### Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:ACM40960/where-is-waldo.git
   cd where-is-waldo

2. **Create a virtual environment** (optional but recommended)

    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt

4. **Install PyTorch** (choose version for your CUDA)

   ```bash
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 # Example for CUDA 11.8
    
---

### Dataset Preparation

1. Download [Where's Waldo dataset](https://www.kaggle.com/datasets/residentmario/wheres-waldo) from Kaggle

2. Create labels using Roboflow and place original images and labels under labelled_data/images and labelled_data/labels respectively.

3. Run preprocessing to tile large pages into 640×640 chips with YOLO-format labels.

    ```bash
    python preprocess.py

By default, preprocess.py writes to ./datasets/train/ (change dest_path to ./datasets/val/ for validation split).