# YOLOv5 Model Export to ONNX

This guide explains how to export the YOLOv5 model to ONNX format. Follow the steps below to complete the export process.

## Cloning the Repository and Installing the Requirements

First, clone the official YOLOv5 repository:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt
```
## Exporting the Model to ONNX

After the dependencies are installed, export the model to ONNX using the command:

```bash
cd /content/yolov5

pip install onnx>=1.10.0
pip install onnxslim
pip install onnxruntime 

python export.py --weights best.pt --img-size 416 416 --batch 1 --include "onnx" --simplify --opset 12

```

Make sure that the --img-size parameter used during the export process matches the image size parameter that was used during training.

_Note: Specifying the appropriate --opset parameter during the ONNX export process is crucial. In my project, when I attempted to export the model to ONNX format without defining the --opset parameter, the model's confidence and accuracy dropped significantly, reaching values as low as 0.0007 - 0.0001, despite having much higher accuracy before the conversion._

_Note: When exporting the official YOLO weight file YOLOv5s.pt on my local machine, there were no issues. However, exporting my custom weight file led to a PosixPath error. I resolved this by switching to Google Colab, where the export process was successful. If you also encounter this issue, you might consider exporting the model on Google Colab. Alternatively, you can find solutions on Stack Overflow regarding how to modify the PosixPath variable._

