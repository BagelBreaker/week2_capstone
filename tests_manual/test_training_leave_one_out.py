import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from get_descriptor import file_descriptors
from vector_db import VectorDatabase

TRAIN_DIR = Path("data/train")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
THRESHOLD = 0.65

def unpack(out):
    if out is None:
        return None
    if len(out) == 2:
        boxes, descriptors = out
        probs = None
    else:
        boxes, probs, descriptors = out
    return boxes, probs, descriptors

entries = []
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

        entries.append({"name": name, "path": str(img_path), "vector": descriptors[0]})

if len(entries) < 2:
    raise RuntimeError("Need at least 2 usable training images for this test.")

correct = 0
attempted = 0

for i, query in enumerate(entries):
    db = VectorDatabase()

    for j, item in enumerate(entries):
        if i == j:
            continue
        db.add(item["vector"], item["name"], {"path": item["path"]})

    result = db.predict(query["vector"], threshold=THRESHOLD)
    pred = result["prediction"]
    sim = result["similarity"]
    ok = pred == query["name"]

    correct += ok
    attempted += 1
    print(f"expected={query['name']} predicted={pred} sim={sim:.3f} ok={ok} path={query['path']}")

print()
print(f"leave-one-out accuracy: {correct}/{attempted} = {correct / attempted:.3f}")
print("skipped:", len(skipped))
for path, reason in skipped[:20]:
    print(" ", reason, path)
