import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from get_descriptor import file_descriptors

TRAIN_DIR = Path("data/train")
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

images = sorted(p for p in TRAIN_DIR.glob("*/*") if p.suffix.lower() in IMG_EXTS)
if not images:
    raise FileNotFoundError("No images found under data/train/*/")

path = images[0]
out = unpack(file_descriptors(path))

print("image:", path)
if out is None:
    print("no face detected")
else:
    boxes, probs, descriptors = out
    print("boxes:", boxes.shape)
    if probs is not None:
        print("probs:", probs)
    print("descriptors:", descriptors.shape)
    print("first descriptor shape:", descriptors[0].shape)
