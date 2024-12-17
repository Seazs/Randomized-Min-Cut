import networkx as nx
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import math
import time 
import random
import pygraphviz as pgv

import cProfile
import pstats

import matplotlib

matplotlib.use('Agg')

class Custom_Graph(nx.MultiGraph):
    def __init__(self):
        super().__init__()
        
    def load_graph(self, filename):
        # load a graph from a txt file where each line is an edge in the form "u v"
        with open (filename, "r") as f:
            for line in f:
                u, v = line.split()
                self.add_edge(int(u), int(v))
    
    def contract_edge(self, u, v):
        """Contract the edge (u, v) in the graph"""
        new_node = f"{u}_{v}"
        self.add_node(new_node)
        
        # Move all edges from u and v to the new node
        for neighbor in list(self.neighbors(u)):
            if neighbor != v:
                self.add_edges_from([(new_node, neighbor)] * self.number_of_edges(u, neighbor))
        for neighbor in list(self.neighbors(v)):
            if neighbor != u:
                self.add_edges_from([(new_node, neighbor)] * self.number_of_edges(v, neighbor))
        
        # Remove u and v and the edge between them
        self.remove_nodes_from([u, v])



    def save_graph(self, filename):
        A = nx.nx_agraph.to_agraph(self)
        A.layout('dot')
        A.draw(filename)
    

def contract_algorithm(graph):
    while len(graph.nodes) > 2:
        u, v, i = random.choice(list(graph.edges))
        graph.contract_edge(u, v)
    return graph

def find_min_cut(graph, num_trials):
    min_cut = float("inf")
    for _ in tqdm(range(num_trials)):
        G = graph.copy()
        G = contract_algorithm(G)
        min_cut = min(min_cut, G.number_of_edges())
    return G, min_cut


if __name__ == "__main__":
    n = 15
    num_trials = n * (n - 1) * int(math.log(n))


    G = Custom_Graph()
    # G.add_edge(1, 2)
    # G.add_edge(1, 2)
    # G.add_edge(2, 3)
    # G.add_edge(3, 4)
    # G.add_edge(4, 1)
    # G.add_edge(1, 3)
    G.load_graph("./data/word_adjacencies.txt")

    G.save_graph("./output/original_graph.png")
    

    min_cut_graph , min_cut = find_min_cut(G, num_trials)
    print(f"Min cut: {min_cut}")
    print(f"Min cut graph: {min_cut_graph.edges}")


