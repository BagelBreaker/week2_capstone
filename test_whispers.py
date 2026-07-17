from whispers import Whispers, cos_dist
import numpy as np
from vector_db import VectorDatabase

testing_segment = 1

if testing_segment == 0:

    vecs = [np.asarray([1,2,3]),
            np.asarray([1,2,3]),
            np.asarray([5,6,7]),
            ]

    names = ["a","b","c"]

    threshold = 0.01

    w = Whispers(vecs,names,threshold)

    if testing_segment == 0:
        w.create_matrix()
        w.create_nodes()
        nodes = w.get_nodes()
        w.train()

elif testing_segment == 1:
    db = VectorDatabase.load("db.pkl")
    w = Whispers(db.vectors,db.names,threshold=0.35)
    w.create_matrix()
    w.create_nodes()
    w.get_plot()
    w.train_sweeps()
    w.print_results()
    w.get_plot()
