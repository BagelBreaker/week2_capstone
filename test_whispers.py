from whispers import Whispers, cos_dist
import numpy as np
from vector_db import VectorDatabase

db = VectorDatabase.load("tests_manual/db.pkl")

test_sweeps = [1,5,10,25,50,100,500,1000]
test_sweeps_2 = range(100,500,20)
num_runs = 50

for sweeps in test_sweeps_2:
    sum = 0
    for run in range(num_runs):
        w = Whispers(db.vectors,db.names,threshold=0.35)
        w.create_matrix()
        w.create_nodes()
        # w.get_plot()
        w.train_sweeps(max_sweeps=sweeps)
        sum += w.get_results()
        # print(w.sorted_images())
        # w.get_plot()
    print(sweeps, " sweeps: ", sum/num_runs)
