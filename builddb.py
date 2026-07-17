from vector_db import VectorDatabase
from get_descriptor import file_descriptors
import os

testing_segment = 1

if testing_segment == 1:
    folder = "data"
    db = VectorDatabase()
    for root, dirs, files in os.walk(folder):
        for image in files:
            image_path = os.path.join(root, image)
            boxes, descriptors = file_descriptors(image_path)
            for descriptor in descriptors: 
                db.add(descriptor, image)
                print("added: ", image)
    db.save("db.pkl")