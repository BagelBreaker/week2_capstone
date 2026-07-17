from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image, ImageDraw

from get_descriptor import file_descriptors
from vector_db import VectorDatabase


DB_PATH = "face_db.pkl"
OUT_DIR = Path("outputs/frontend")
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Face Recognizer", layout="wide")

st.title("Face Recognizer")

threshold = st.sidebar.slider("Recognition threshold", 0.30, 0.95, 0.65, 0.01)
min_prob = st.sidebar.slider("Minimum face detection probability", 0.50, 1.00, 0.95, 0.01)
min_area_frac = st.sidebar.slider("Minimum face size", 0.000, 0.100, 0.010, 0.001)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if not Path(DB_PATH).exists():
    st.error(f"Could not find {DB_PATH}. Run builddb.py first.")
    st.stop()

db = VectorDatabase.load(DB_PATH)

if uploaded is None:
    st.info("Upload a photo to run recognition.")
    st.stop()

suffix = Path(uploaded.name).suffix.lower()

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    img_path = Path(tmp.name)

img = Image.open(img_path).convert("RGB")
img_w, img_h = img.size

out = file_descriptors(img_path)

if out is None:
    st.warning("No faces detected.")
    st.image(img, caption="Uploaded image", use_container_width=True)
    st.stop()

boxes, probs, descriptors = out

items = []

for i, (box, prob, desc) in enumerate(zip(boxes, probs, descriptors)):
    x1, y1, x2, y2 = box
    area_frac = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

    if prob < min_prob or area_frac < min_area_frac:
        continue

    result = db.predict(desc, threshold=threshold)
    pred = result["prediction"]
    sim = result["similarity"]

    items.append({
        "face_index": len(items),
        "box": box,
        "prediction": pred,
        "similarity": sim,
        "probability": float(prob),
        "area_frac": float(area_frac),
        "top_matches": result["results"],
    })

draw = ImageDraw.Draw(img)

for item in items:
    i = item["face_index"]
    x1, y1, x2, y2 = item["box"]
    pred = item["prediction"]
    sim = item["similarity"]

    label = f"Face {i}: {pred} ({sim:.2f})"

    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    text_x = x1
    text_y = max(0, y1 - 18)

    bbox = draw.textbbox((text_x, text_y), label)
    draw.rectangle(bbox, fill="white")
    draw.text((text_x, text_y), label, fill="red")

out_path = OUT_DIR / f"{Path(uploaded.name).stem}_annotated.jpg"
img.save(out_path)

left, right = st.columns([2, 1])

with left:
    st.subheader("Annotated image")
    st.image(img, use_container_width=True)

with right:
    st.subheader("Detected faces")

    if not items:
        st.warning("Faces were detected, but all were filtered out.")
    else:
        rows = []

        for item in items:
            rows.append({
                "Face": item["face_index"],
                "Prediction": item["prediction"],
                "Similarity": round(item["similarity"], 3),
                "Detection prob": round(item["probability"], 3),
                "Face size": round(item["area_frac"], 4),
            })

        st.table(rows)

        st.subheader("Top matches")

        for item in items:
            st.write(f"**Face {item['face_index']}**")

            top_rows = [
                {
                    "Name": match["name"],
                    "Similarity": round(match["similarity"], 3),
                }
                for match in item["top_matches"][:5]
            ]

            st.table(top_rows)

st.success(f"Saved annotated image to {out_path}")