import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from get_descriptor import file_descriptors
from vector_db import VectorDatabase

DB_PATH = "face_vectors.pkl"
GROUP_DIR = Path("data/group_photos")
OUT_DIR = Path("outputs/group_results")
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

db = VectorDatabase.load(DB_PATH)
OUT_DIR.mkdir(parents=True, exist_ok=True)
images = sorted(p for p in GROUP_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)

if not images:
    raise FileNotFoundError("No group photos found under data/group_photos/")

for img_path in images:
    out = unpack(file_descriptors(img_path))
    print("\nimage:", img_path)

    if out is None:
        print("no faces detected")
        continue

    boxes, probs, descriptors = out
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for k, (box, descriptor) in enumerate(zip(boxes, descriptors)):
        result = db.predict(descriptor, threshold=THRESHOLD)
        pred = result["prediction"]
        sim = result["similarity"]
        prob = None if probs is None else float(probs[k])

        x1, y1, x2, y2 = map(float, box)
        label = f"{pred} {sim:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill="red")

        print(f" face {k}: pred={pred} sim={sim:.3f} detect_prob={prob} box={box}")
        print("  top matches:")
        for match in result["results"][:3]:
            print(f"   {match['name']} {match['similarity']:.3f}")

    out_path = OUT_DIR / f"{img_path.stem}_labeled.jpg"
    img.save(out_path)
    print("saved labeled image:", out_path)
