# 🐟 Comprehensive Computer Vision System: Fish Classification, Detection & Segmentation
**Course:** CPCS-432: Artificial Intelligence (II)  
**Semester:** 1st Semester, 2026  
**Institution:** King Abdulaziz University  
**Department:** Computer Science  

## 📌 Project Overview
This project consolidates advanced computer vision techniques to solve the problem of fish species analysis. It integrates a wide range of methodologies—from classical machine learning with handcrafted features to state-of-the-art Deep Learning models—into a unified framework.

The system is designed to perform three core tasks:
1.  **Image Classification:** Identifying fish species using both classical (HOG+ANN/SVM) and deep learning (CNNs, Transfer Learning) approaches.
2.  **Object Detection & Segmentation:** Localizing fish in images and generating pixel-perfect segmentation masks.
3.  **Decision-Level Fusion:** Implementing a Stacking Ensemble to combine the strengths of heterogeneous models for maximum accuracy.

## 🚀 Key Features & Methodologies

### 1. Classical Machine Learning
* **Feature Extraction:** Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP).
* **Classifiers:** Artificial Neural Networks (ANN), SVM, KNN.

### 2. Deep Learning Models
* **Custom CNNs:** Designed from scratch, tested with regularization techniques like Dropout.
* **Transfer Learning:** Fine-tuning pre-trained architectures like **MobileNetV2** for robust performance on limited data.

### 3. Advanced Fusion Strategy
* **Technique:** Stacked Generalization (Stacking).
* **Mechanism:** A meta-learner (Logistic Regression) aggregates probability scores from the best classical (HOG+ANN) and deep learning models to improve final classification accuracy and robustness.

### 4. Detection & Segmentation
* **Models:** Implementation of architectures like YOLO / Mask R-CNN for precise object localization and segmentation.

### 5. Deployment
* **Web Application:** A fully functional web interface allows users to upload images and visualize classification results in real-time.

