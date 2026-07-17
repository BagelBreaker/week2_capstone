from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from feedback import collect_feedback
from get_descriptor import file_descriptors
from vector_db import VectorDatabase

PKL_PATH = "face_db.pkl"
FILE_PATH = "data/group_photos/group1.jpg"
OUT_DIR = Path("outputs/group_results")
THRESHOLD = 0.65


def main():
    db = VectorDatabase.load(PKL_PATH)
    out = file_descriptors(FILE_PATH)

    if out is None:
        print("No faces in image")
        return

    boxes, probs, descriptors = out
    predictions = []

    for i, (box, prob, desc) in enumerate(zip(boxes, probs, descriptors)):
        result = db.predict(desc, threshold=THRESHOLD)
        prediction = result["prediction"]
        predictions.append(prediction)

        print(f"Face {i}: {prediction} sim={result['similarity']:.3f} prob={prob:.3f}")

        if prediction == "unknown":
            collect_feedback(result, desc, db, metadata={"source": FILE_PATH, "box": box.tolist()})

    db.save(PKL_PATH)

    img = Image.open(FILE_PATH).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(img)

    for box, label in zip(boxes, predictions):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, fontsize=10)

    ax.axis("off")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / Path(FILE_PATH).name
    plt.savefig(out_path, bbox_inches="tight")
    print("saved labeled image:", out_path)
    plt.show()


if __name__ == "__main__":
    main()
