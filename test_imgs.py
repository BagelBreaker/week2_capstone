from get_descriptor import file_descriptors
import pickle
from vector_db import VectorDatabase
from feedback1 import collect_feedback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
PKL_PATH = "face_db.pkl"
FILE_PATH = "data/Mixed_groups/lebron_serena.jpg"

db = VectorDatabase.load(PKL_PATH)


boxes, descriptors = file_descriptors(FILE_PATH)
if descriptors is None:
    print("No faces in image")
else:
    predictions = []
    for i, desc in enumerate(descriptors):
        result = db.predict(desc)
        prediction = result['prediction']
        print(f"Face {i}: {prediction}")
        predictions.append(prediction)
        if prediction == "unknown":
            collect_feedback(prediction, desc, db)
    #plot imgs
    fig, ax = plt.subplots()
    img = Image.open(FILE_PATH).convert("RGB")
    ax.imshow(img)
    for box, label in zip(boxes,predictions):
        xmin, xmax, ymin, ymax = box
        width = xmax-xmin
        height = ymax-ymin
        rect = patches.Rectangle((xmin,ymin), width, height)
        ax.add_patch(rect)
        ax.text(xmin,ymin-5, label, color='red', fontsize=10)
    ax.axis('off')
    plt.show()
        




