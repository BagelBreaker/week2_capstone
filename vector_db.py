import pickle
from pathlib import Path

import numpy as np


class VectorDatabase:
    def __init__(self):
        self.vectors = []
        self.names = []
        self.metadata = []

    def add(self, vector, name, metadata=None):
        self.vectors.append(np.asarray(vector, dtype=np.float32).ravel())
        self.names.append(str(name))
        self.metadata.append(metadata or {})

    def add_many(self, vectors, name, metadata_list=None):
        metadata_list = metadata_list or [{} for _ in range(len(vectors))]
        for vector, metadata in zip(vectors, metadata_list):
            self.add(vector, name, metadata)

    def labels(self):
        return sorted(set(self.names))

    def as_matrix(self):
        return np.vstack(self.vectors) if self.vectors else np.empty((0, 0), dtype=np.float32)

    def save(self, path):
        with open(Path(path), "wb") as f:
            pickle.dump({"vectors": self.vectors, "names": self.names, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path):
        with open(Path(path), "rb") as f:
            data = pickle.load(f)

        db = cls()
        db.vectors = data["vectors"]
        db.names = data["names"]
        db.metadata = data["metadata"]
        return db

    def query(self, vector, top_k=5):
        if not self.vectors:
            return []

        vector = np.asarray(vector, dtype=np.float32).ravel()
        matrix = self.as_matrix()

        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        vector = vector / (np.linalg.norm(vector) + 1e-12)

        sims = matrix @ vector
        idxs = np.argsort(sims)[::-1][:top_k]

        return [
            {"name": self.names[i], "similarity": float(sims[i]), "metadata": self.metadata[i]}
            for i in idxs
        ]

    def predict(self, vector, threshold=0.65, top_k=5):
        results = self.query(vector, top_k)

        if not results:
            return {"prediction": "unknown", "similarity": 0.0, "margin": 0.0, "results": []}

        best = results[0]
        second = results[1]["similarity"] if len(results) > 1 else 0.0
        margin = best["similarity"] - second
        prediction = best["name"] if best["similarity"] >= threshold else "unknown"

        return {
            "prediction": prediction,
            "similarity": best["similarity"],
            "margin": float(margin),
            "results": results,
        }
