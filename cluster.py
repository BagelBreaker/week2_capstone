from whispers import Whispers
from vector_db import VectorDatabase

# file dedicated to sorting images

DB_PATH = "tests_manual/db.pkl"
THRESHOLD = 0.35
MAX_SWEEPS = 280

db = VectorDatabase.load(DB_PATH)
w = Whispers(db.vectors, db.names, threshold=THRESHOLD)
w.create_matrix()
w.create_nodes()
w.train_sweeps(max_sweeps=MAX_SWEEPS)

# shows the plot
w.get_plot()

# auto sorts images
print(w.get_results())
print(w.sorted_images())