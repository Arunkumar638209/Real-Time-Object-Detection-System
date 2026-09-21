import cv2
import numpy as np
import torch

def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Apply Non-Max Suppression to filter overlapping boxes.
    """
    # YOLO handles NMS internally, but custom implementation if needed
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

def augment_image(image, rotation=True, flip=True, contrast=True):
    """
    Apply augmentations: rotation, flipping, contrast adjustment.
    """
    if flip and np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    if rotation:
        angle = np.random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
    if contrast:
        alpha = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return image

def resize_image(image, size=(640, 640)):
    """
    Resize image to specified size.
    """
    return cv2.resize(image, size)

# Other utilities as needed
