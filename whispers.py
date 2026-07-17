import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List
import random
from collections import Counter

class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        """
        Parameters
        ----------
        ID : int
            A unique identifier for this node. Should be a
            value in [0, N-1], if there are N nodes in total.

        neighbors : Sequence[int]
            The node-IDs of the neighbors of this node.

        descriptor : numpy.ndarray
            The shape-(512,) descriptor vector for the face that this node corresponds to.

        truth : Optional[str]
            If you have truth data, for checking your clustering algorithm,
            you can include the label to check your clusters at the end.
            If this node corresponds to a picture of Ryan, this truth
            value can just be "Ryan"

        file_path : Optional[str]
            The file path of the image corresponding to this node, so
            that you can sort the photos after you run your clustering
            algorithm
        """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path
    
    def get_neighbors(self):
        return self.neighbors
    
    def set_label(self,label):
        self.label = label

    def get_label(self):
        return self.label

def plot_graph(graph, adj):
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.
    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.
    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot. This is simple a tuple of the nodes in the graph.
        Each element should be an instance of the `Node`-class.

    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot."""

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax

def cos_dist(a, b):
    return 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

class Whispers:
    def __init__(self, vectors, names, threshold):
        self.threshold = threshold
        self.vectors = vectors
        self.names = names
        self.num_nodes = len(vectors)

        self.adj_mat = np.zeros([self.num_nodes,self.num_nodes])
        self.nodes = []

    def create_matrix(self):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes - i):
                vec_a = self.vectors[i]
                vec_b = self.vectors[j]
                if (i != j) and (cos_dist(vec_a, vec_b) <= self.threshold):
                    self.adj_mat[i,j] = 1
                    self.adj_mat[j,i] = 1


    def create_nodes(self):
        for i in range(self.num_nodes):
            self.nodes.append(Node(i,
                                   np.where(self.adj_mat[i] == 1)[0],
                                   self.vectors[i],
                                   self.names[i],
                                   ))
            
    def get_adj_mat(self):
        return self.adj_mat
    
    def get_nodes(self):
        return self.nodes
    
    def get_plot(self):
        plot_graph(self.nodes,self.adj_mat)
        plt.show()

    def whispers_step(self):
        selected_node = random.choice(self.nodes)
        old_label = selected_node.get_label()
        counts = Counter()
        neighbors = selected_node.get_neighbors()
        if len(neighbors) > 0:
            for neighbor in neighbors:
                counts[self.nodes[neighbor].get_label()] += 1
            max_count = max(counts.values())
            new_label = random.choice([id for id, count in counts.items() if count == max_count])
            selected_node.set_label(new_label)
        else:
            new_label = old_label
        return old_label, new_label

    def train(self, convergence_threshold=10, stopping_threshold=1000): # only run this once
        label_counts = Counter()
        total_labels = len(self.nodes)
        for node in self.nodes:
            label_counts[node.get_label()] += 1
        idle_iterations = 0
        total_iterations = 0
        while idle_iterations < convergence_threshold and total_iterations < stopping_threshold:
            old_label, new_label = self.whispers_step()
            label_counts[old_label] -= 1
            label_counts[new_label] += 1
            if label_counts[old_label] == 0:
                total_labels -= 1
                idle_iterations = 0
            else:
                idle_iterations += 1
            total_iterations += 1
            self.get_plot()
        
            
