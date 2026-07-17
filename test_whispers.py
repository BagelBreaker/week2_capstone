from whispers import Whispers, cos_dist
import numpy as np

testing_segment = 0

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
