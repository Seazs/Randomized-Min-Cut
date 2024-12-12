import random
import time
import math
import pygraphviz as pgv
import copy
from tqdm import tqdm

class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v
    # define how to print the edge
    def __repr__(self):
        return f"{self.u} -- {self.v}"

class Graph:
    def __init__(self):
        self.V = 0
        self.E = 0
        self.edges = []
        self.edges_copy = []
        random.seed(time.time())
    
    def load_graph(self, filename):
        # load a graph from a txt file where each line is an edge in the form "u v"
        with open(filename, "r") as f:
            for line in f:
                u, v = line.split()
                self.add_edge(Edge(int(u), int(v)))
        self.reset_graph()
            


    def get_edges(self):
        return self.edges


    def reset_graph(self):

        # reset the value of V
        self.V = 0
        for edge in self.edges:
            self.V = max(self.V, edge.u + 1, edge.v + 1)
        self.E = len(self.edges)

    def add_edge(self, edge):
        self.edges.append(edge)
        self.E = len(self.edges)
        self.V = max(self.V, edge.u + 1, edge.v + 1)

    def print_graph(self):
        for edge in self.edges:
            print(f"{edge.u} -- {edge.v}")

    def contract_edge(self, u, v):
        """Contract the edge (u, v) in the graph"""
        
        for edge in self.edges_copy: # update the edges
            if edge.u == v:
                edge.u = u
            if edge.v == v:
                edge.v = u

        self.edges_copy = [edge for edge in self.edges_copy if edge.u != edge.v] # remove self loops
        self.V -= 1 # update the number of vertices

    def contract_algorithm(self):

        self.edges_copy = copy.deepcopy(self.edges) # copy the edges (deep copy is required as we will modify the edges)
        while self.V > 2: # contract the edges until we have only 2 vertices
            edge = random.choice(self.edges_copy)
            self.contract_edge(edge.u, edge.v)
        contracted_edges = self.edges_copy
        self.reset_graph()
        return contracted_edges
        

    def create_graph_png(self, filename):
        graph = pgv.AGraph(strict=False, directed=False)
        for edge in self.edges:
            graph.add_edge(edge.u, edge.v)
        graph.draw(filename, prog='dot', format='png')

def create_random_graph(V, E):
    graph = Graph()
    unique_edges = set()
    # Ensure the graph is connected by creating a spanning tree first
    for i in range(1, V):
        u = random.randint(0, i - 1)
        v = i
        unique_edges.add((u, v))
    # Add remaining edges randomly
    while len(unique_edges) < E:
        u = random.randint(0, V - 1)
        v = random.randint(0, V - 1)
        if u != v:
            edge = tuple(sorted((u, v)))
            unique_edges.add(edge)
    for u, v in unique_edges:
        graph.add_edge(Edge(u, v))
    return graph

def find_min_cut(graph, num_trials):
    min_cut_edges = []
    min_cut = len(graph.get_edges())
    for _ in range(num_trials):
        contracted_edges = graph.contract_algorithm()
        if len(contracted_edges) < min_cut:
            min_cut = len(contracted_edges)
            min_cut_edges = contracted_edges
    return min_cut, min_cut_edges

def compute_theorical_error_probability(n, num_trials):
    return ((1 - (2 / (n * (n - 1)))) ** num_trials)

def compute_empirical_success_probability(graph : Graph, n_test, num_trials, min_cut):
    """
    Compute the empirical success probability of the algorithm
    
    Parameters:
    graph : Graph
        The graph to test
    n_test : int
        The number of tests to perform
    num_trials : int
        The number of trials to perform for each test
    min_cut : int
        The minimum cut of the graph"""

    fail = 0
    for _ in tqdm(range(n_test)):
        min_cut_trial, min_cut_edges = find_min_cut(graph, num_trials)
        if min_cut_trial != min_cut:
            fail += 1
    return fail / n_test


if __name__ == "__main__":
    n = 30
    m = 100
    num_trials = n * (n - 1) * int(math.log(n))

    print(f"n: {n}, m: {m}, num_trials: {num_trials}")

    graph = create_random_graph(n, m)
    print(graph.get_edges())
    #graph = Graph()
    # graph.load_graph("word_adjacencies.txt")
    graph.create_graph_png("initial_graph.png")
    print(graph.E, graph.V)
    

    min_cut, min_cut_edges = find_min_cut(graph, 10)
    print(f"Minimum cut: {min_cut}")
    print(min_cut_edges)

    pb = compute_theorical_error_probability(n, num_trials)
    print(f"Theorical error probability: {pb}")

    n_test = 500
    pe = compute_empirical_success_probability(graph, n_test, num_trials, min_cut)
    print(f"Empirical error probability: {pe}")

    

