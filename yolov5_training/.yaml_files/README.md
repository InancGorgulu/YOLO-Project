## YAML Configuration Files for YOLO Training

This folder contains the necessary `.yaml` files for configuring and training your YOLO models. These configuration files are essential for setting up your dataset paths, model architecture, and training parameters.

**Note:** Please ensure to customize the `.yaml` files according to your project requirements, including paths and model configurations.

### data.yaml

This file specifies the dataset paths and class information for the training process. It includes:
- **train**: Path to the training images.
- **val**: Path to the validation images.
- **test**: Path to the test images (if available).
- **nc**: Number of classes to detect.
- **names**: List of class names.

_Ensure to update these paths according to your dataset location on Google Colab or your local system._

### custom_yolov5s.yaml

This file configures the training parameters for the YOLOv5S model. It is based on the official `yolov5s.yaml` file provided by Ultralytics, with modifications for:
- **Number of classes (`nc`)**: Adjusted to match the number of classes in your dataset.
- **Model architecture**: Other parameters, such as `depth_multiple` and `width_multiple`, can be modified to control the model's complexity and size.

_If you want to customize the model further, you can modify additional parameters such as `anchors` or `backbone` layers._
