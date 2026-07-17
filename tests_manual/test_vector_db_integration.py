import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from get_descriptor import file_descriptors
from vector_db import VectorDatabase


TRAIN_DIR = Path("data/train")
DB_PATH = "face_vectors.pkl"

db = VectorDatabase()

for person_dir in sorted(TRAIN_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    name = person_dir.name

    for img_path in sorted(person_dir.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        out = file_descriptors(img_path)

        if out is None:
            print("no face:", img_path)
            continue

        boxes, descriptors = out

        if len(descriptors) != 1:
            print("skipping multi-face image:", img_path)
            continue

        db.add(descriptors[0], name, {"path": str(img_path)})

db.save(DB_PATH)

print("saved:", DB_PATH)
print("vectors:", len(db.vectors))
print("people:", sorted(set(db.names)))