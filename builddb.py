from pathlib import Path
from vector_db import VectorDatabase

import numpy as np

from get_descriptor import file_descriptors
import os

testing_segment = 2

if testing_segment == 1:
    folder = "data/test"
    db = VectorDatabase()
    for root, dirs, files in os.walk(folder):
        for image in files:
            try:
                image_path = os.path.join(root, image)
                boxes, descriptors = file_descriptors(image_path)
                for descriptor in descriptors: 
                    db.add(descriptor, image[:-4])
                    print("added: ", image)
            except Exception as e:
                print("skipped: ", image)
    db.save("db.pkl")
elif testing_segment == 2:
    folder = "data/test"
    db = VectorDatabase()
    for root, dirs, files in os.walk(folder):
        for image in files:
            image_path = os.path.join(root, image)
            boxes, probs, descriptors = file_descriptors(image_path)
            for descriptor in descriptors: 
                db.add(descriptor, image[:-4])
                print("added: ", image)
    db.save("db.pkl")