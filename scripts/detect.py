import cv2
import torch
from ultralytics import YOLO
import os

def real_time_detection(model_path, save_crops=False, export_onnx=False):
    """
    Perform real-time object detection using webcam.
    """
    # Load model
    model = YOLO(model_path)

    # Export to ONNX if requested
    if export_onnx:
        model.export(format='onnx')

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame, conf=0.5)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Save cropped images if enabled
        if save_crops:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(f'crops/{box.cls.item()}_{len(os.listdir("crops"))}.jpg', crop)

        # Display
        cv2.imshow('Real-Time Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use pre-trained model for demo
    model_path = 'yolov8n.pt'  # Pre-trained model
    real_time_detection(model_path, save_crops=True, export_onnx=True)
