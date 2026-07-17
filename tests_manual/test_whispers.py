from whispers import Whispers, cos_dist
import numpy as np

testing_segment = 1

vecs = [np.asarray([1,2,3]),
        np.asarray([1,2,3]),
        np.asarray([5,6,7]),
        ]

names = ["a","b","c"]

threshold = 0.01

w = Whispers(vecs,names,threshold)

if testing_segment == 1:
    w.create_matrix()
    w.create_nodes()
    nodes = w.get_nodes()
    w.whispers_step()
    w.get_plot()

elif testing_segment == 2:
    print(cos_dist(vecs[0],vecs[2]))
    print(vecs[0].shape)