from pathlib import Path

import numpy as np

from get_descriptor import file_descriptors
from vector_db import VectorDatabase

TRAIN_DIR = Path("data/train")
DB_PATH = "face_db.pkl"
MIN_FACE_PROB = 0.95
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def unpack(out):
    if out is None:
        return None
    if len(out) == 2:
        boxes, descriptors = out
        probs = np.ones(len(descriptors))
    else:
        boxes, probs, descriptors = out
    return boxes, probs, descriptors


def main():
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
                skipped.append((img_path, "no face/bad image"))
                continue

            boxes, probs, descriptors = out
            keep = probs >= MIN_FACE_PROB
            boxes, descriptors = boxes[keep], descriptors[keep]

            if len(descriptors) != 1:
                skipped.append((img_path, f"{len(descriptors)} faces"))
                continue

            db.add(descriptors[0], name, {"path": str(img_path), "box": boxes[0].tolist()})
            print("added:", name, img_path.name)

    db.save(DB_PATH)
    print("saved:", DB_PATH)
    print("vectors:", len(db.vectors))
    print("people:", db.labels())
    print("skipped:", len(skipped))
    for path, reason in skipped:
        print(" ", reason, path)


if __name__ == "__main__":
    main()
