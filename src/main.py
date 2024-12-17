import random
import time
import math
import pygraphviz as pgv
import copy
import itertools

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

    def contract_algorithm(self, t=2):

        self.edges_copy = copy.deepcopy(self.edges) # copy the edges (deep copy is required as we will modify the edges)
        while self.V > t: # contract the edges until we have only 2 vertices
            edge = random.choice(self.edges_copy)
            self.contract_edge(edge.u, edge.v)
        contracted_edges = self.edges_copy
        self.reset_graph()
        return contracted_edges
    
    def is_connected(self, edges):
        """Check if the graph is connected after removing the given edges"""
        remaining_edges = [edge for edge in self.edges if edge not in edges]
        if not remaining_edges:
            return False

        # Create a set of connected components
        components = []
        for edge in remaining_edges:
            u_component = None
            v_component = None
            for component in components:
                if edge.u in component:
                    u_component = component
                if edge.v in component:
                    v_component = component
            if u_component and v_component:
                if u_component != v_component:
                    u_component.update(v_component)
                    components.remove(v_component)
            elif u_component:
                u_component.add(edge.v)
            elif v_component:
                v_component.add(edge.u)
            else:
                components.append(set([edge.u, edge.v]))

        # Check if all vertices are in a single component
        all_vertices = set()
        for component in components:
            all_vertices.update(component)
        return len(all_vertices) == self.V

    def brutforce(self):
        """Brutforce algorithm to find the minimum cut by trying every subset of edges"""
        min_cut = self.E
        min_cut_edges = []
        print("°°°")
        print(self.V)
        print(self.E)
        print(self.edges)
        for r in range(1, self.E + 1):
            for edges_subset in itertools.combinations(self.edges, r):
                if not self.is_connected(edges_subset):
                    if len(edges_subset) < min_cut:
                        min_cut = len(edges_subset)
                        min_cut_edges = edges_subset


        return min_cut_edges
    
    def contract_to_size(self, H, t):
        while H.V > t:
            edge = random.choice(H.edges)
            H.contract_edge(edge.u, edge.v)
        return H
    
    def fast_cut_algorithm(self):
        n = self.V
        if n <= 6:
            return self.brutforce()
        
        t = math.ceil(1+ n / math.sqrt(2))

        H1 = copy.deepcopy(self)
        H2 = copy.deepcopy(self)

        

        
            

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

def compute_theorical_error_probability_contract(n):
    return (2 / (n * (n - 1)))

def compute_theorical_error_probability_fast_cut(n):
    return 1/math.log(n)

def compute_empirical_success_probability(graph : Graph, n_test, min_cut, algo = "contract"):
    """
    Compute the empirical success probability of the algorithm when run once
    
    Parameters:
    graph : Graph
        The graph to test
    n_test : int
        The number of tests to perform
    min_cut : int
        The minimum cut of the graph"""

    success = 0
    for _ in tqdm(range(n_test)):
        if algo == "contract":
            contracted_edges = graph.contract_algorithm()
        else:
            contracted_edges = graph.fast_cut_algorithm()
        if len(contracted_edges) == min_cut:
            success += 1
    return success / n_test

def compute_theorical_error_probability_multiple_trials(n, num_trials):
    return ((1 - (2 / (n * (n - 1)))) ** num_trials)

def compute_empirical_success_probability_multiple_trials(graph : Graph, n_test, num_trials, min_cut):
    """
    Compute the empirical success probability of the algorithm when run num_trials iterations in a row
    
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
    n = 15
    m = (3/4)*(n*(n-1)/2) - 1
    num_trials = n * (n - 1) * int(math.log(n))
    #num_trials = 100

    print(f"n: {n}, m: {m}, num_trials: {num_trials}")

    graph = create_random_graph(n, m)
    print(graph.get_edges())
    # graph = Graph()
    # graph.load_graph("./data/word_adjacencies.txt")
    graph.create_graph_png("./output/initial_graph.png")
    print(graph.E, graph.V)
    

    min_cut, min_cut_edges = find_min_cut(graph, num_trials)
    print(f"Minimum cut: {min_cut}")
    print(min_cut_edges)

    # theorical_success_rate = compute_theorical_error_probability_contract(n)
    # print(f"Theorical success rate when run once for contract algo: {theorical_success_rate}")

    # empirical_success_rate = compute_empirical_success_probability(graph, 500, min_cut)
    # print(f"Empirical success rate when run once: {empirical_success_rate}")

    # pb = compute_theorical_error_probability_multiple_trials(n, num_trials)
    # print(f"Theorical error probability when run {num_trials} times : {pb}")

    # n_test = 500
    # pe = compute_empirical_success_probability_multiple_trials(graph, n_test, num_trials, min_cut)
    # print(f"Empirical error probability when run {num_trials} times : {pe}")

    # theorical_success_rate_fast_cut = compute_theorical_error_probability_fast_cut(n)
    # print(f"Theorical success rate when run once for fast cut algo: {theorical_success_rate_fast_cut}")

    empirical_success_rate_fast_cut = compute_empirical_success_probability(graph, 500, min_cut, "fast_cut")
    print(f"Empirical success rate when run once for fast cut algo: {empirical_success_rate_fast_cut}")

    

