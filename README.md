
# Where's Waldo? – Automated Visual Search using YOLOv11

![Find Waldo Illustration](/assets/images/1.jpg "Find Waldo Illustration")

## 📜 Overview

This project implements an automated visual search system to detect Waldo in dense illustrations using the YOLOv11 object detection model. Each illustration is split into 640×640 sub-images, with bounding box annotations for Waldo when present. The pipeline is optimized for small-object detection and rare-class scenarios, using tailored hyperparameters and targeted data augmentation.

---

## 📂 Project Structure

```bash
where-is-waldo/
├── assets/ # assets for readme
│ └── images/ # images for README.md
│
└── data.yaml # yaml file (to be referenced by train.py)
├── datasets/ # train and val datasets (populated from preprocess.py ignored in git due to huge dataset)
│ ├── train/ # 70% of dataset
│ │  ├── images/ # jpg files
│ │  └── labels/ # txt files
│ └── val/  # 20 % of dataset
│    ├── images/ # jpg files
│    └── labels/ # txt files
│
├── labelled_data/ # raw dataset from kaggle ignored in git upload due to huge dataset
│ ├── images/ # high-res original images
│ └── labels/ # resp labels of each img from Roboflow
|
│── models/ #ignored in git upload 
│ ├── yolo11n_custom.pt # pretrained weights (model configuration)
|
├── tests/ # 10% testing dataset ignored in git upload due to huge dataset
│    ├── input/ # input data for test
│    └── output/ # output of test
│
├── waldoData/ # 10% testing dataset ignored in git upload due to huge dataset
│    ├── Clean/ # input data for test
│    │ ├── ClearedWaldos/ # bg imgs with no waldo
│    │ └── OnlyWaldoHeads/ # only waldo heads
│    ├── NotWaldo/ # imgs populate from generateData.py
│    └── Waldo/ # imgs populate from generateData.py
│
├── .gitignore
├── customLib.py # python utility functions
├── generateData.py # generate training data****
├── inference.py # predict on test data
├── preprocess.py # preprocess data****
├── README.md 
├── requirements.txt
├── train.py # python script to train data
```

---

## 🚀 Features

1. Tiling of large illustrations into YOLO-compatible sub-images

2. Annotation handling for images using [RoboFLow](https://app.roboflow.com/)

3. YOLOv11n small-object optimization

4. Steps: preprocessing → training → evaluation → inference

![Project Structure](/assets/images/waldov2.png "Project Structure")

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
    
5. **Define all User Defined Functions** (define and run all functions to be used in below steps)

   ```bash
    python customLib.py
    ```
    
---

### Dataset Preparation

1. Download [Where's Waldo dataset](https://www.kaggle.com/datasets/residentmario/wheres-waldo) from Kaggle

2. Create labels using Roboflow and place original images and labels under labelled_data/images and labelled_data/labels respectively.

3. Run preprocessing to tile large pages into 640×640 chips with YOLO-format labels.

    ```bash
    python preprocess.py

By default, preprocess.py writes to ./datasets/train/ (change dest_path to ./datasets/val/ for validation split).

4. (OPTIONAL) Generate extra Training Data - Use onlyWaldoHeads to paste on clear backgrounds with some rotations into 640×640 chips. Images with Waldo will populate in waldoData/Waldo folder. 

    4.1 Repeat step 2 to create new labels

    4.2 Append datasets/train and datasets/val accordingly.

    ```bash
    python generateData.py

## Model Configuration
![YOLO Model Architecture](/assets/images/YOLOv11.webp "YOLOv11 Architecture")

In our model, we use YOLOv11 (You Only Look Once version 11), a state-of-the-art real-time object detection architecture. YOLOv11 enhances detection accuracy and speed by incorporating lightweight backbone networks, improved anchor-free detection heads, and dynamic label assignment strategies. For our configuration, we fine-tuned YOLOv11 with custom dataset-specific hyperparameters, including image size, batch size, learning rate, and training epochs, ensuring optimal performance for the target detection task.

### Training :
The YOLOv11 model was trained for object detection using the lightweight **yolo11n.pt** backbone. Training was performed for 500 epochs with a batch size of 10 and an image size of 640×640 pixels. A weight decay of 0.0005 was applied to regularize the model and reduce overfitting. The dataset path and class configuration were defined in the **data.yaml** file, and training output (including weights and logs) was saved automatically by Ultralytics.

```bash
    python train.py
```


### Validation :
Model performance was evaluated on the validation set using the best model checkpoint saved during training. The validation metrics (such as mAP, precision, and recall) were calculated based on the annotations defined in the dataset.

```bash
    python validate.py
```
### Testing:

For final testing and inference, the best trained model was used to make predictions on the test images. A confidence threshold of 0.573 was set to filter detections. The predictions were saved for further visualization and review.

```bash
    python inference.py
```

If the performance metrics are satisfactory, the best-performing model weights are exported and saved for deployment.
If not, the training process is revisited with adjusted hyperparameters for further improvement.

### Result : There he is !
The model is now able to detect Waldo accurately within the images, demonstrating the effectiveness of our object detection pipeline.

![Found Waldo](/assets/images/Found_Waldo.png "Found Waldo")