
# Where's Waldo? – Automated Visual Search using YOLOv11

![Find Waldo Illustration](/assets/images/18.jpg "Find Waldo Illustration")

📜 Overview
This project implements an automated visual search system to detect Waldo in high-resolution illustrations using the YOLOv11n object detection model. Each illustration is split into 640×640 sub-images, with bounding box annotations for Waldo when present. The pipeline is optimized for small-object detection and rare-class scenarios, using tailored hyperparameters and targeted data augmentation.