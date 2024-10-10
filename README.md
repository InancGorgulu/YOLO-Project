# YOLO-Project: Comprehensive Guide to Training and Inference

Welcome to the YOLO-Project repository! This repository is designed to provide a comprehensive guide for training the YOLO (You Only Look Once) algorithm and utilizing the trained models in C++ for real-time object detection and classification.

## Overview
YOLO is a state-of-the-art, real-time object detection system that is known for its speed and accuracy. This repository is divided into two main sections:

### YOLO Training (YOLOv5) 
This section covers the complete process of training the YOLO model using the YOLOv5 framework. You can find detailed instructions, data preparation techniques, and model training scripts tailored to your specific dataset and use case.

**Note:** *The entire tutorial is documented mostly for the **Google Colab** environment for faster prototyping.*

### C++ Model Inference 
This section demonstrates how to integrate and deploy the trained YOLO model within a C++ environment.

## Repository Structure
The repository is organized into separate folders, each dedicated to different aspects of model training and deployment:

### yolov5_training/
This folder contains all the resources needed for training YOLO models using YOLOv5. 

### cpp_inference/
This folder provides the tools and source code necessary for integrating and running the trained YOLO model in a C++ environment with OpenCV library.

## Getting Started
### 1. Training YOLO Models
To train your custom YOLO model, navigate to the `yolov5_training/` directory and follow the steps outlined in the README.md. This section includes:

* Setting up the Google Colab environment with the necessary dependencies. 
* Preparing your dataset and defining the structure in data.yaml.
* Running the training script with customized parameters.

### 2. Using Trained Models in C++
Once your model is trained and validated, you can use it in a C++ environment. The `cpp_inference/` directory contains everything you need to get started, including:

* Converting the PyTorch(_YOLOv5_) model to ONNX.
* Compiling and running the inference code to test the model with images.

## License

This project is licensed under the MIT License.

## Contact
For questions or further assistance, feel free to reach out via GitHub Issues or contact the repository maintainers.
