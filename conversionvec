import os
import torch
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "images")

mtcnn = MTCNN(image_size=160)
model = InceptionResnetV1(pretrained="vggface2").eval()
descriptors = {}


for person in os.listdir(IMAGE_FOLDER):

    person_folder = os.path.join(IMAGE_FOLDER, person)

    if not os.path.isdir(person_folder):
        continue

    descriptors[person] = {}

    print("Processing person:", person)

    for filename in os.listdir(person_folder):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(person_folder, filename)
            print("  Image:", filename)
            img = Image.open(path).convert("RGB")
            face = mtcnn(img)
            if face is None:
                print("  No face found")
                continue

            face = face.unsqueeze(0)

            with torch.no_grad():
                descriptor = model(face)

            descriptors[person][filename] = (descriptor.squeeze().numpy())
with open("descriptors.pkl", "wb") as f:
    pickle.dump(descriptors, f)

print("Finished")
print("Saved", len(descriptors), "people")
