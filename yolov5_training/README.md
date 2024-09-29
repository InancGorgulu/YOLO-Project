# YOLOv5 Training Project

This repository contains all the necessary components for training a custom YOLOv5 model using Google Colab. The project is organized into various folders, each serving a specific purpose. Below is a detailed explanation of each directory and its contents.

## Repository Structure

- ### `colaboratory/`
  Contains the (`YOLO_training.ipynb`) that can be used directly in Google Colab. It provides step-by-step instructions for setting up the environment, loading the dataset, configuring the model, and training.

- ### `configurations/`
  - **`custom_yolov5.yaml`**: This file defines the custom architecture for the YOLOv5 model, such as the number of layers, channels, and anchor boxes.
  - **`data.yaml`**: Specifies the paths to the training, validation, and test datasets. It also includes class names and the number of classes.

- ### `dataset/`
  Contains (`README.md`) related to the dataset used in this project. This document describes the dataset structure, source information, and preprocessing steps applied before training.

- ### `pretrained_weights/`
  Contains the pre-trained weights (`yolov5s.pt`) for the YOLOv5 model. These weights can be used as a starting point for transfer learning, which can speed up the training process and improve accuracy.

### Training Configuration Tips

- Ensure that the `nc` parameter in `custom_yolov5s.yaml` matches the number of classes in your dataset defined in `data.yaml` and check the path definitions.
- Adjust the `batch size` and `epochs` according to Google Colab GPU capabilities (or local GPU) to prevent memory issues.
- Customize the `anchors` parameter in the model configuration file to better suit the object sizes in your dataset.

## Example Training Command

To start training using the provided datasets and configuration files, execute the following command in your Colab notebook:

```python
!python train.py --img 416 --batch 16 --epochs 100 --data /content/data.yaml --cfg /content/yolov5/models/custom_yolov5s.yaml --weights '' --name yolov5s_results --cache

--img 416:
Specifies the input image size. In this case, images will be resized to 416x416 pixels before being fed into the model.
Usage: This parameter controls the dimensions of the input images. Larger sizes typically lead to better accuracy but require more computational resources.

--batch 16:
Sets the batch size, meaning the number of images that will be processed together in a single batch during training.
Usage: A larger batch size can speed up training but also requires more memory. Smaller batch sizes may lead to better generalization.

--epochs 100:
Defines the number of training epochs. An epoch is one complete pass through the entire training dataset.
Usage: More epochs allow the model to learn better, but too many can lead to overfitting.

--data /content/data.yaml:
Specifies the path to the data.yaml file, which contains information about the dataset, such as the path to the training, validation, and test images, as well as class names and the number of classes.
Usage: This file is crucial for configuring the training data paths and class definitions.

--cfg /content/yolov5/models/custom_yolov5s.yaml:
Provides the path to the custom model configuration file (.yaml) for YOLOv5. This file contains the architecture details of the model such as the number of layers, channels, and anchor boxes.
Usage: Use this parameter to define a custom model structure or modify an existing model to suit your data.

--weights '':
Specifies the path to the weights file. Leaving it empty ('') means that no pre-trained weights are loaded and the model will be trained from scratch.
Usage: You can use a path to pre-trained weights (e.g., yolov5s.pt) to perform transfer learning, or leave it empty to start with random weights.

--name yolov5s_results:
Sets the name of the experiment. The results will be saved in a folder named yolov5s_results.
Usage: Use a descriptive name to organize your experiments, making it easier to track and compare different training runs.

--cache:
Caches images and labels in memory for faster training.
Usage: Speeds up training by reducing the disk I/O for loading images. Useful when the dataset is small enough to fit into memory.