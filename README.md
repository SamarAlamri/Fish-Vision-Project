# 🐟 Comprehensive Computer Vision System: Fish Classification, Detection & Segmentation
**Course:** CPCS-432: Artificial Intelligence (II)  
**Semester:** 1st Semester, 2026  
**Institution:** King Abdulaziz University  
**Department:** Computer Science  

## 📌 Project Overview
This project consolidates a massive array of computer vision experiments into a unified framework for fish species analysis. The system integrates contributions from all group members, covering **Classical Machine Learning**, **Deep Learning (CNNs)**, **Transfer Learning**, **Object Detection**, and **Instance Segmentation**.

## ⚠️ IMPORTANT: Download Trained Models
Due to GitHub's file size limits (maximum 100MB per file), the trained model files (weights) are hosted externally. 

**👉 [CLICK HERE TO DOWNLOAD MODELS FROM GOOGLE DRIVE](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

**Instructions:**
1. Download the `.zip` file from the link above.
2. Extract the contents.
3. Place the `.h5`, `.pth`, and `.pt` files into the `Web_Application/models/` folder before running the program.

---

## 🚀 Key Features & Models Implemented

### 1. Classical Machine Learning
Feature extraction techniques combined with strong classifiers (**ANN, KNN, SVM**):
* **Texture & Pattern:** LBP (Local Binary Patterns), GLCM (Gray-Level Co-occurrence Matrix).
* **Shape & Geometry:** HOG (Histogram of Oriented Gradients), HuMoments.
* **Keypoints:** ORB (Oriented FAST and Rotated BRIEF).

### 2. Deep Learning 
Four custom Convolutional Neural Networks designed from scratch:
* **CNN Model 1 & 2:** Baseline architectures for feature learning.
* **CNN Model 3 & 4:** Advanced variations with Dropout and Batch Normalization.

### 3. Transfer Learning (Pre-trained Models)
Fine-tuned state-of-the-art architectures for robust classification:
* **MobileNetV2** (Optimized for speed).
* **ResNet18** (Residual learning).
* **InceptionV3** (Multi-scale processing).
* **EfficientNet-B0** (Efficiency & Accuracy balance).

### 4. Object Detection
Detects fish and draws a bounding box around them:
* **YOLO Series:** YOLOv5, YOLOv8n, YOLOv11, and Custom trained versions.
* **Transformer-based:** RT-DETR (Real-Time DEtection TRansformer).

### 5. Image Segmentation
Pixel-level mask generation for precise fish boundaries:
* **Semantic/Instance Segmentation:** Mask R-CNN, U-Net, DeepLabV, FCN (Fully Convolutional Network).
* **Modern Architectures:** YOLOv8-Seg, Custom SegNet.

### 6. Decision-Level Fusion (The "Manager")
* **Technique:** Stacked Generalization (Stacking).
* **Role:** Combines the predictions of the best-performing models (e.g., ANN-HOG + MobileNetV2 + Custom CNN) using a Logistic Regression meta-learner to achieve **100% Validation Accuracy**.

---

## 📂 Repository Structure

```text
Fish-Vision-Project/
│
├── 📂 Project_Report           # Full documentation (Methodology, Experiments, Results)
│   └── Comprehensive_Computer_Vision_Report.pdf
│
├── 📂 Web_Application          # Source code for the deployment interface
│   ├── app.py                  # Main application script
│   ├── templates/              # HTML frontend
│   └── saved_models/           # PLACE DOWNLOADED MODELS HERE
│
├── 📂 Notebooks                # Experimental Code
│   ├── 📂 Final_Project_Fusion # The unified Fusion Model code
│   └── 📂 Student_Assignments  # Individual contributions (Member 1, 2, 3...)
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
🛠️ How to Run the System
Prerequisites
Ensure you have Python 3.8+ installed.

1. Installation
Clone the repository:

Bash

git clone [https://github.com/SamarAlamri/Fish-Vision-Project.git](https://github.com/SamarAlamri/Fish-Vision-Project.git)
cd Fish-Vision-Project
Install dependencies:

Bash

pip install -r requirements.txt
2. Setup Models
Download the models from the Google Drive link at the top of this file.

Move them into the Web_Application/saved_models/ directory.

3. Run the Web Application
Navigate to the folder and start the server:

Bash

cd Web_Application
python app.py
Open your browser and visit: http://127.0.0.1:5000

📊 Results Summary
Best Classical Model: ANN + HOG features (>90% accuracy).

Best Deep Model: MobileNetV2 (~96% accuracy).

Fusion Model: The Stacking Ensemble achieved 100% accuracy on the validation set.

👥 Group Members
[Member 1 Name] (ID: XXXXXX)

[Member 2 Name] (ID: XXXXXX)

[Member 3 Name] (ID: XXXXXX)

[Member 4 Name] (ID: XXXXXX)

Developed for CPCS-432 Course Project.
