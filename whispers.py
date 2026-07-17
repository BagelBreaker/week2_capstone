from collections import Counter
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Node:
    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        self.id = ID
        self.label = ID
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor
        self.truth = truth
        self.file_path = file_path

    def get_neighbors(self):
        return self.neighbors

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label


def cos_dist(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return 1 - (a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)


def plot_graph(graph, adj):
    g = nx.Graph()
    g.add_nodes_from(range(len(graph)))
    g.add_edges_from(zip(*np.where(np.triu(adj, 1) > 0)))

    pos = nx.spring_layout(g)
    labels = sorted(set(node.label for node in graph))
    colors = list(iter(cm.tab20b(np.linspace(0, 1, len(labels)))))
    color_map = dict(zip(labels, colors))
    node_colors = [color_map[node.label] for node in graph]

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=node_colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax


class Whispers:
    def __init__(self, vectors, names, threshold):
        self.threshold = threshold
        self.vectors = [np.asarray(v, dtype=np.float32).ravel() for v in vectors]
        self.names = names
        self.num_nodes = len(vectors)
        self.adj_mat = np.zeros((self.num_nodes, self.num_nodes))
        self.nodes = []

    def create_matrix(self):
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if cos_dist(self.vectors[i], self.vectors[j]) <= self.threshold:
                    self.adj_mat[i, j] = 1
                    self.adj_mat[j, i] = 1

    def create_nodes(self):
        self.nodes = [
            Node(i, np.where(self.adj_mat[i] == 1)[0], self.vectors[i], self.names[i])
            for i in range(self.num_nodes)
        ]

    def get_adj_mat(self):
        return self.adj_mat

    def get_nodes(self):
        return self.nodes

    def get_plot(self):
        fig, ax = plot_graph(self.nodes, self.adj_mat)
        plt.show()
        return fig, ax

    def whispers_step(self):
        node = random.choice(self.nodes)
        old_label = node.get_label()
        neighbors = node.get_neighbors()

        if len(neighbors) == 0:
            return old_label, old_label

        counts = Counter(self.nodes[n].get_label() for n in neighbors)
        max_count = max(counts.values())
        new_label = random.choice([label for label, count in counts.items() if count == max_count])
        node.set_label(new_label)
        return old_label, new_label

    def train(self, max_steps=1000, patience=100, plot=False):
        if not self.nodes:
            self.create_matrix()
            self.create_nodes()

        unchanged = 0
        for step in range(max_steps):
            old, new = self.whispers_step()
            unchanged = unchanged + 1 if old == new else 0

            if plot and step % 50 == 0:
                self.get_plot()

            if unchanged >= patience:
                break

        return self.clusters()

    def clusters(self):
        groups = {}
        for node in self.nodes:
            groups.setdefault(node.label, []).append(node.id)
        return groups
