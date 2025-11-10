# Real-Time Object Detection System

This project implements a real-time object detection system using YOLOv8, with support for training on COCO or custom datasets, and real-time detection via webcam.

## Features

- **Model**: YOLOv8 (Ultralytics)
- **Training**: Transfer learning on COCO or custom datasets
- **Preprocessing**: Image resizing, augmentations (rotation, flipping, contrast)
- **Evaluation**: mAP, IoU
- **Real-Time Detection**: OpenCV webcam, bounding boxes, confidence scores
- **Optimization**: Export to ONNX
- **Extras**: Save cropped detections, training curves, model export

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download COCO dataset or prepare custom dataset in YOLO format (images + labels).

3. Create data YAML file, e.g., `data/coco.yaml`:
   ```
   train: data/coco/train2017
   val: data/coco/val2017
   nc: 80
   names: ['person', 'bicycle', ...]  # COCO classes
   ```

## Usage

### Training
```
python scripts/train.py
```
Adjust parameters in the script for custom datasets.

### Real-Time Detection
```
python scripts/detect.py
```
Press 'q' to quit. Set `save_crops=True` to save detected objects.

### Custom Dataset
- Prepare images in `data/custom/images/`
- Annotations in YOLO format in `data/custom/labels/`
- Update YAML accordingly.

## Notes
- CPU-based since CUDA not available.
- For GPU, install CUDA and change device to 'cuda'.
- Ensure webcam is available for detection.
