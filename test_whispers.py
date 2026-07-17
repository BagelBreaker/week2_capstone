from whispers import Whispers, cos_dist
import numpy as np
from vector_db import VectorDatabase

db = VectorDatabase.load("db.pkl")
w = Whispers(db.vectors,db.names,threshold=0.35)
w.create_matrix()
w.create_nodes()
w.get_plot()
w.train_sweeps()
w.print_results()
w.get_plot()
