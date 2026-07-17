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

    def get_truth(self):
        return self.truth


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
            for j in range(i+ 1, self.num_nodes):
                vec_a = self.vectors[i]
                vec_b = self.vectors[j]
                if (i != j) and (cos_dist(vec_a, vec_b) <= self.threshold):
                    self.adj_mat[i,j] = 1
                    self.adj_mat[j,i] = 1


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

    def train(self, convergence_threshold=10, stopping_threshold=10000): # only run this once
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
            print("num labels: ", total_labels)
            print("old: ", old_label, "new: ", new_label)
        
        self.get_plot()

    def whispers_sweep(self):
        nodes_random = self.nodes.copy()
        random.shuffle(nodes_random)
        for selected_node in nodes_random:
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

    def train_sweeps(self, max_sweeps=1000):
        for sweep in range(max_sweeps):
            self.whispers_sweep()
    
    def print_results(self):
        label_counts = Counter()
        for node in self.nodes:
            label_counts[node.get_label()] += 1
        print("total clusters: ", len(label_counts.keys()))
        for label in label_counts.keys():
            print("cluster: ", self.names[label])
    
    def sorted_images(self):
        label_counts = Counter()
        sorted = {}
        for node in self.nodes:
            label_counts[node.get_label()] += 1
        for label in label_counts.keys():
            sorted[self.nodes[label].get_truth()] = []
        for node in self.nodes:
            sorted[self.nodes[node.get_label()].get_truth()].append(node.get_truth())
        return sorted
    
    def get_results(self):
        label_counts = Counter()
        for node in self.nodes:
            label_counts[node.get_label()] += 1
        return len(label_counts.keys())
            

    
    


        
            

        
            
