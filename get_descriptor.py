from camera import take_picture
from facenet_models import FacenetModel
from PIL import Image
import numpy as np
model = FacenetModel()

def file_descriptors(file_pth):
    img = np.array(Image.open(file_pth))
    boxes, probabilities, landmarks = model.detect(img)
    if boxes is None or len(boxes) == 0:
        return None
    descriptors = model.compute_descriptors(img, boxes)

    return boxes, descriptors
    

def camera_descriptors():
    img = take_picture()
    boxes, probabilities, landmarks = model.detect(img)
    if boxes is None or len(boxes) == 0:
        return None
    descriptors = model.compute_descriptors(img, boxes)
    return boxes, descriptors

