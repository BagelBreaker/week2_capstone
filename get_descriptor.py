from pathlib import Path

import cv2

import numpy as np
from facenet_models import FacenetModel
from PIL import Image, UnidentifiedImageError

model = FacenetModel()


def _load_img(path):
    with Image.open(Path(path)) as img:
        return np.array(img.convert("RGB"))


def file_descriptors(file_path):
    try:
        img = _load_img(file_path)
    except (UnidentifiedImageError, OSError) as e:
        print("bad image skipped:", file_path)
        print(" ", e)
        return None

    boxes, probs, landmarks = model.detect(img)
    if boxes is None or len(boxes) == 0:
        return None

    descriptors = model.compute_descriptors(img, boxes)
    return boxes, probs, descriptors


def camera_descriptors():
    from camera import take_picture

    img = take_picture()
    boxes, probs, landmarks = model.detect(img)
    if boxes is None or len(boxes) == 0:
        return None

    descriptors = model.compute_descriptors(img, boxes)
    return boxes, probs, descriptors
