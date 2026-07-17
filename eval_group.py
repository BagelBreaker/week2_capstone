from pathlib import Path
import csv

from get_descriptor import file_descriptors
from vector_db import VectorDatabase


PKL_PATH = "face_db.pkl"
GROUP_DIR = Path("data/group_photos")
TRUTH_PATH = Path("annotations/group_truth.csv")

THRESHOLD = 0.65
MIN_PROB = 0.95
MIN_AREA_FRAC = 0.01


def filtered_faces(img_path):
    from PIL import Image

    out = file_descriptors(img_path)
    if out is None:
        return []

    boxes, probs, descs = out
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    kept = []

    for box, prob, desc in zip(boxes, probs, descs):
        x1, y1, x2, y2 = box
        area_frac = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

        if prob >= MIN_PROB and area_frac >= MIN_AREA_FRAC:
            kept.append((box, prob, desc))

    return kept


truth = {}

with open(TRUTH_PATH, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        image = row["image"]
        face_index = int(row["face_index"])
        true_label = row["true_label"]

        truth[(image, face_index)] = true_label


db = VectorDatabase.load(PKL_PATH)

total = 0
correct = 0

known_total = 0
known_correct = 0

unknown_total = 0
unknown_correct = 0

false_known = 0
mistakes = []

for image_name in sorted(set(image for image, _ in truth)):
    img_path = GROUP_DIR / image_name
    faces = filtered_faces(img_path)

    for face_index, (box, prob, desc) in enumerate(faces):
        key = (image_name, face_index)

        if key not in truth:
            continue

        true_label = truth[key]
        result = db.predict(desc, threshold=THRESHOLD)
        pred = result["prediction"]
        sim = result["similarity"]

        total += 1

        if pred == true_label:
            correct += 1
        else:
            mistakes.append((image_name, face_index, true_label, pred, sim))

        if true_label == "unknown":
            unknown_total += 1
            if pred == "unknown":
                unknown_correct += 1
            else:
                false_known += 1
        else:
            known_total += 1
            if pred == true_label:
                known_correct += 1


print()
print("Evaluation results")
print("------------------")
print(f"overall: {correct}/{total} = {correct / total:.3f}")

if known_total:
    print(f"known faces: {known_correct}/{known_total} = {known_correct / known_total:.3f}")

if unknown_total:
    print(f"unknown faces: {unknown_correct}/{unknown_total} = {unknown_correct / unknown_total:.3f}")

print("false-known errors:", false_known)

print()
print("mistakes:")
for image_name, face_index, true_label, pred, sim in mistakes:
    print(f"{image_name} Face {face_index}: true={true_label}, pred={pred}, sim={sim:.3f}")