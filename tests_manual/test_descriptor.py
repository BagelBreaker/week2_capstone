import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from get_descriptor import file_descriptors

path = "data/train/02_samaltman/samaltman1.jpg"

out = file_descriptors(path)

if out is None:
    print("no face detected")
else:
    boxes, descriptors = out
    print("boxes:", boxes.shape)
    print("descriptors:", descriptors.shape)
    print("first descriptor shape:", descriptors[0].shape)