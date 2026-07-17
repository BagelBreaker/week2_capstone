from pathlib import Path
import csv
import shutil

import numpy as np
from PIL import Image

from get_descriptor import file_descriptors
from vector_db import VectorDatabase


PKL_PATH = "face_db.pkl"
EVAL_DIR = Path("data/test")
OUT_DIR = Path("outputs/eval_sorted")
CSV_PATH = Path("outputs/eval_results.csv")

THRESHOLD = 0.65
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

db = VectorDatabase.load(PKL_PATH)


def largest_face(boxes, probs, descs):
    best_i = None
    best_area = -1

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            best_i = i

    if best_i is None:
        return None

    return boxes[best_i], probs[best_i], descs[best_i]


rows = []
total = correct = 0
known_total = known_correct = 0
unknown_total = unknown_correct = 0
false_known = 0
skipped = 0

for true_dir in sorted(EVAL_DIR.iterdir()):
    if not true_dir.is_dir():
        continue

    true_label = true_dir.name

    for img_path in sorted(true_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        out = file_descriptors(img_path)

        if out is None:
            pred = "no_face"
            sim = 0.0
            skipped += 1
        else:
            boxes, probs, descs = out
            face = largest_face(boxes, probs, descs)

            if face is None:
                pred = "no_face"
                sim = 0.0
                skipped += 1
            else:
                box, prob, desc = face
                result = db.predict(desc, threshold=THRESHOLD)
                pred = result["prediction"]
                sim = result["similarity"]

        is_correct = pred == true_label

        total += 1
        correct += is_correct

        if true_label == "unknown":
            unknown_total += 1
            unknown_correct += is_correct
            if pred not in ["unknown", "no_face"]:
                false_known += 1
        else:
            known_total += 1
            known_correct += is_correct

        rows.append({
            "image": str(img_path),
            "true_label": true_label,
            "prediction": pred,
            "similarity": round(sim, 4),
            "correct": is_correct,
        })

        pred_dir = OUT_DIR / pred
        pred_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, pred_dir / img_path.name)

        mark = "OK" if is_correct else "BAD"
        print(f"{mark} {img_path} true={true_label} pred={pred} sim={sim:.3f}")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image", "true_label", "prediction", "similarity", "correct"],
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("Evaluation results")
print("------------------")
print(f"overall: {correct}/{total} = {correct / total:.3f}")

if known_total:
    print(f"known faces: {known_correct}/{known_total} = {known_correct / known_total:.3f}")

if unknown_total:
    print(f"unknown faces: {unknown_correct}/{unknown_total} = {unknown_correct / unknown_total:.3f}")

print("false-known errors:", false_known)
print("skipped/no face:", skipped)
print("csv saved:", CSV_PATH)
print("sorted outputs:", OUT_DIR)