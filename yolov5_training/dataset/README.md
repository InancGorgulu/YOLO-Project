# Dataset Overview

This repository contains the datasets and configuration files necessary for training and evaluating YOLO models. Below is a summary of the included datasets and instructions for obtaining and using additional datasets via Roboflow.

_Here is the URL of Drone dataset that used for this training project:_
[Drone Dataset](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav/data)


## Dataset Summary

I have divided the '.txt' and '.jpg' files that represented in 'drone_dataset_yolo' file into the 'labels' and 'images' folders. Also, decreased the number of the images for faster prototyping in my project.

# File Hierarchy

- **train/**: Contains training data _(Approximately %70 of the dataset)_. 
  - **images/**: The directory where all training images are stored.
  - **labels/**: The directory containing annotation files corresponding to the training images.
  
- **valid/**: Contains validation data _(Approximately %10-15 of the dataset)_.
  - **images/**: The directory where all validation images are stored.
  - **labels/**: The directory containing annotation files corresponding to the validation images.

- **test/**: Contains test data _(Approximately %10-15 of the dataset)_.
  - **images/**: The directory where all test images are stored.
  - **labels/**: The directory containing annotation files corresponding to the test images.

## Obtaining Datasets from Roboflow

Roboflow is a powerful tool for creating, annotating, and managing datasets for machine learning projects. You can easily download pre-annotated datasets or create your own. Here’s how you can obtain and use datasets from Roboflow:

1. **Create a Roboflow Account**: Sign up for a free [Roboflow account](https://roboflow.com/).
2. **Select or Upload Your Dataset**:
   - Browse through public datasets or upload your own images. 
   - Annotate the images using the Roboflow interface or utilize pre-annotated datasets.
3. **Generate Dataset and Export**:
   - Configure the dataset by selecting the train-test split ratio.
   - Choose the export format as `YOLO`.
4. **Download the Dataset**:

_Use the Roboflow API for downloading the dataset in your environment:_ 

- Press the "Show download code" to get API_KEY provided by Roboflow to download your dataset in the appropriate format in a cloud machine such as Google Colab.

 _or_
  
_Download directly the required dataset on your local machine:_
     
- Press the "Download zip to computer" to get your dataset in the appropriate format.


### Using of the Roboflow API
```ipython
!pip install roboflow 
# This is an IPython command that can be run in environments like Google Colab or Jupyter Notebook. To run it in your terminal, remove the '!' prefix. The other codes included in below can be run in your Python Shell or Python Script.

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
# You have to use your private API Key to provide access to your Roboflow account in your code.

project = rf.workspace("my-workshop-xttfp").project("my-project-name-zzqav")
version = project.version(3)

dataset = version.download("yolov5")
```            

## Uploading Datasets to Google Colab

To use the dataset in Google Colab, follow these steps:

1. **Upload Dataset to Colab**:
   - If your dataset is small, you can directly upload it to the Colab environment using the left sidebar’s file upload option.
   - For larger datasets, you can use the `gdown` library to download datasets directly from Google Drive:
     ```python
     !pip install gdown
     !gdown https://drive.google.com/uc?id=your_file_id
     ```
   - Alternatively, as showed above, you can download it via Roboflow API.

2. **Mount Google Drive** (if the dataset is stored in your Google Drive):
   - Mount Google Drive to access files directly from your Drive.
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Navigate to your dataset directory in Drive:
     ```python
     dataset_path = '/content/drive/MyDrive/your_dataset_folder'
     ```

3. **Configure Paths in `data.yaml`**:
   - Update the `data.yaml` file with the correct paths in your Colab environment:
     ```yaml
     train: /content/your_dataset_folder/train/images
     val: /content/your_dataset_folder/valid/images
     test: /content/your_dataset_folder/test/images

     nc: 1  # Number of classes
     names: ['class_name']  # Class names
     ```

