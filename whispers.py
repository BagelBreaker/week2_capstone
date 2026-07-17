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


def cos_dist(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return 1 - a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)


def plot_graph(graph, adj):
    g = nx.Graph()
    g.add_nodes_from(range(len(graph)))
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    pos = nx.spring_layout(g)
    labels = sorted(set(node.label for node in graph))
    colors = list(cm.tab20b(np.linspace(0, 1, len(labels))))
    color_map = dict(zip(labels, colors))
    node_colors = [color_map[node.label] for node in graph]

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=node_colors)
    nx.draw_networkx_edges(g, pos=pos, ax=ax, edgelist=g.edges())
    return fig, ax


class Whispers:
    def __init__(self, vectors, names=None, threshold=0.35):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.names = list(names) if names is not None else [None] * len(vectors)
        self.threshold = threshold
        self.num_nodes = len(self.vectors)
        self.adj_mat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.nodes = []

    def create_matrix(self):
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dist = cos_dist(self.vectors[i], self.vectors[j])
                if dist <= self.threshold:
                    weight = 1 - dist
                    self.adj_mat[i, j] = weight
                    self.adj_mat[j, i] = weight

    def create_nodes(self):
        self.nodes = [
            Node(i, np.where(self.adj_mat[i] > 0)[0], self.vectors[i], self.names[i])
            for i in range(self.num_nodes)
        ]

    def run(self, max_iter=20):
        if not self.adj_mat.any():
            self.create_matrix()
        if not self.nodes:
            self.create_nodes()

        for _ in range(max_iter):
            changed = False
            order = list(range(self.num_nodes))
            random.shuffle(order)

            for i in order:
                votes = {}

                for j in self.nodes[i].neighbors:
                    label = self.nodes[j].label
                    votes[label] = votes.get(label, 0) + self.adj_mat[i, j]

                if not votes:
                    continue

                new_label = max(votes, key=votes.get)

                if new_label != self.nodes[i].label:
                    self.nodes[i].label = new_label
                    changed = True

            if not changed:
                break

        return self.clusters()

    def clusters(self):
        groups = {}

        for node in self.nodes:
            groups.setdefault(node.label, []).append({
                "id": node.id,
                "name": node.truth,
                "file_path": node.file_path,
            })

        return groups

    def get_adj_mat(self):
        return self.adj_mat

    def get_plot(self):
        fig, ax = plot_graph(self.nodes, self.adj_mat)
        plt.show()
        return fig, ax
