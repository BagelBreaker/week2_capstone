import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from get_descriptor import file_descriptors
from vector_db import VectorDatabase

TRAIN_DIR = Path("../data/test")
DB_PATH = "face_vectors.pkl"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def unpack(out):
    if out is None:
        return None
    if len(out) == 2:
        boxes, descriptors = out
        probs = None
    else:
        boxes, probs, descriptors = out
    return boxes, probs, descriptors

db = VectorDatabase()
skipped = []

for person_dir in sorted(TRAIN_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    name = person_dir.name
    for img_path in sorted(person_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        out = unpack(file_descriptors(img_path))
        if out is None:
            skipped.append((str(img_path), "no face"))
            continue

        boxes, probs, descriptors = out
        if len(descriptors) != 1:
            skipped.append((str(img_path), f"{len(descriptors)} faces"))
            continue

        prob = None if probs is None else float(probs[0])
        db.add(descriptors[0], name, {"path": str(img_path), "face_prob": prob})

if not db.vectors:
    raise RuntimeError("No usable face vectors were added. Check data/train and image quality.")

db.save(DB_PATH)

print("saved:", DB_PATH)
print("vectors:", len(db.vectors))
print("people:", sorted(set(db.names)))
print("skipped:", len(skipped))
for path, reason in skipped[:20]:
    print(" ", reason, path)
