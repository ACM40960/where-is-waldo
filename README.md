
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
├── datasetV3/ # train and val datasets (populated from preprocess.py ignored in git due to huge dataset)
│ ├── 3-Fold_Cross-val/ 
│ │  ├── split_1/ # train val split 1
│ │  ├── split_2/ # train val split 2
│ │  └── split_3/ # train val split 3
│ ├── test/  # 10% testing dataset ignored in git upload due to huge dataset
│ │  ├── images/ 
│ │  └── labels/ 
│ └── train_val/ 
│    ├── images/ 
│    └── labels/ 
│
├── labelled_data/ # raw dataset from kaggle ignored in git upload due to huge dataset
│ ├── images/ # high-res original images
│ └── labels/ # resp labels of each img from Roboflow
│
├── waldoData/ # data used in generating more dataset
│    ├── Clean/ 
│    │ ├── ClearedWaldos/ # bg imgs with no waldo
│    │ └── OnlyWaldoHeads/ # only waldo heads
│    ├── NotWaldo/ # imgs populate from generateData.py
│    └── Waldo/ # imgs populate from generateData.py
├── .gitignore
├── customLib.py # python utility functions
├── data.yaml # yaml file (to be referenced by train.py)
├── generateData.py # generate training data
├── preprocess.py # preprocess data****
├── README.md 
├── requirements.txt
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
    
---

### Dataset Preparation

1. Download [Where's Waldo dataset](https://www.kaggle.com/datasets/residentmario/wheres-waldo) from Kaggle

2. Create labels using Roboflow and place original images and labels under labelled_data/images and labelled_data/labels respectively.

3. Run preprocessing to tile large pages into 640×640 chips with YOLO-format labels.

    ```bash
    python preprocess.py

By default, preprocess.py writes to ./datasetV3/train_val/ (dir contains both train and validation set).

4. (OPTIONAL) Generate extra Training Data - Use onlyWaldoHeads to paste on ClearedWaldos backgrounds with some rotations into 640×640 chips. Images with Waldo will populate in waldoData/Waldo folder. 

    4.1 Repeat step 2 to create new labels

    4.2 Append datasetV3/train_val accordingly.

    ```bash
    python generateData.py

OR

1. Directly download the refined dataset from [here - datasetV3](https://www.kaggle.com/datasets/bakshi15/waldo-dataset-v3) this has separate test directory and a combined train+validation directory(which gets randomly divided in cross validation step)

## Model Configuration

![YOLO Model Architecture](/assets/images/YOLOv11.webp "YOLOv11 Architecture")

In our model, we use YOLOv11 (You Only Look Once version 11), a state-of-the-art real-time object detection architecture. YOLOv11 enhances detection accuracy and speed by incorporating lightweight backbone networks, improved anchor-free detection heads, and dynamic label assignment strategies. For our configuration, we fine-tuned YOLOv11 with custom dataset-specific hyperparameters, including image size, batch size, learning rate, and training epochs, ensuring optimal performance for the target detection task.

## Refer wald0-search.iypnb file for next steps:

### Cross Validation :

The notebook uses K-Fold cross-validation to split the dataset:

n_splits: Number of folds (e.g., 3 or 5)

random_state: Ensures reproducibility

This allows the model to train and validate across multiple subsets of data.

Here we applied 3-fold cross validation

### Training :

Use a deep learning model (e.g., YOLO from ultralytics) for training.

For each fold:
train_data: Data for model training
val_data: Data for validation
Repeat for each fold to improve generalization

The YOLOv11 model was trained for object detection using the lightweight **yolo11n.pt** backbone. Training was performed for 100 epochs with a batch size of 16 and an image size of 640×640 pixels.The dataset path and class configuration were defined in the respective **data.yaml** file, and training output (including weights and logs) were saved automatically by Ultralytics.


### Validation :
Each split Model performance was evaluated on the validation set using the best model checkpoint saved during training. The validation metrics (such as mAP, precision, and recall) were calculated based on the annotations defined in the dataset.

### Testing:

For final testing and inference, the best trained model was used to make predictions on the test images. The predictions were saved for further visualization and review.

predictions = model.predict(test_images)
for img, pred in zip(test_images, predictions):
    display(pred.show())

model.predict(): Returns bounding boxes or class predictions


### Result : There he is !
The model is now able to detect Waldo accurately within the images, demonstrating the effectiveness of our object detection pipeline.

![Found Waldo](/assets/images/Found_Waldo.png "Found Waldo")