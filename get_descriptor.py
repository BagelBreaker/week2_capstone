from pathlib import Path

import numpy as np
from facenet_models import FacenetModel
from PIL import Image

model = FacenetModel()


def _load_img(path):
    with Image.open(Path(path)) as img:
        return np.array(img.convert("RGB"))


def file_descriptors(file_path):
    img = _load_img(file_path)
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
