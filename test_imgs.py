from pathlib import Path

from get_descriptor import file_descriptors
from vector_db import VectorDatabase
from feedback import collect_feedback

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


PKL_PATH = "face_db.pkl"
GROUP_DIR = Path("data/group_photos")
OUT_DIR = Path("outputs/group_results")

ASK_FEEDBACK = False
THRESHOLD = 0.65
SHOW_UNKNOWN_LABELS = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

db = VectorDatabase.load(PKL_PATH)

image_paths = [
    p for p in sorted(GROUP_DIR.iterdir())
    if p.suffix.lower() in IMG_EXTS
]

if not image_paths:
    print("No group photos found in", GROUP_DIR)
    raise SystemExit

for img_path in image_paths:
    print()
    print("processing:", img_path)

    out = file_descriptors(img_path)

    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(img)

    if out is None:
        print("  no faces detected")
        ax.text(
            10, 20, "No faces detected",
            color="red",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    else:
        boxes, probs, descriptors = out
        face_items = []

        for i, (box, prob, desc) in enumerate(zip(boxes, probs, descriptors)):
            result = db.predict(desc, threshold=THRESHOLD)
            pred = result["prediction"]
            sim = result["similarity"]

            print(f"  Face {i}: {pred} sim={sim:.3f} prob={prob:.3f}")

            face_items.append({
                "index": i,
                "box": box,
                "prob": prob,
                "desc": desc,
                "result": result,
                "pred": pred,
                "sim": sim,
            })

            x1, y1, x2, y2 = box
            show_label = pred != "unknown" or SHOW_UNKNOWN_LABELS
            label = f"Face {i}: {pred}"

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)

            if show_label:
                ax.text(
                    x1,
                    max(y1 - 6, 10),
                    label,
                    color="red",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )

    ax.axis("off")

    out_path = OUT_DIR / f"{img_path.stem}_labeled.jpg"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print("  saved:", out_path)

    if ASK_FEEDBACK and out is not None:
        for item in face_items:
            if item["pred"] == "unknown":
                print()
                print(f"Feedback for {img_path.name}, Face {item['index']}")
                collect_feedback(
                    item["result"],
                    item["desc"],
                    db,
                    metadata={
                        "path": str(img_path),
                        "face_index": item["index"],
                        "prob": float(item["prob"]),
                    },
                )

        db.save(PKL_PATH)