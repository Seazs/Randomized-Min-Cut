import random
import itertools
import copy
import pygraphviz as pgv
import math
import networkx as nx

from functools import lru_cache


class Graph:
    def __init__(self, graph_type="random", V=10, E=20):
        self.init()
        if graph_type == "random":
            self.create_random_graph(V, E)
        elif graph_type == "complete":
            self.create_complete_graph(V)
        elif graph_type == "cycle":
            self.create_cycle_graph(V)
        elif graph_type == "star":
            self.create_star_graph(V)
        
    
    def init(self):
        self.edges = []
        self.init_edges = []
        self._E = 0
        self._V = 0
    
    # this 2 properties allow to automatically update the number of edges and vertices when they are called
    @property
    def V(self):
        vertices = set(itertools.chain.from_iterable(self.edges))
        self._V = len(vertices)
        return self._V
    
    @property
    def E(self):
        self._E = len(self.edges)
        return self._E
    
    def remember_edges(self):
        self.init_edges = copy.deepcopy(self.edges)
        
    def load_graph(self, filename):
        # load a graph from a txt file where each line is an edge in the form "u v"
        with open (filename, "r") as f:
            for line in f:
                u, v = line.split()
                self.add_edge(int(u), int(v))
        self.init_edges = copy.deepcopy(self.edges)
        
    def add_edge(self, u, v):
        self.edges.append((u, v))
        
    def reset_graph(self):
        self.edges = copy.deepcopy(self.init_edges)
        
    def contract_edge(self, u, v):
        """Contract the edge (u, v) in the graph"""
        new_edges = []
        for edge in self.edges:
            if edge[0] == v:
                edge = (u, edge[1])
            if edge[1] == v:
                edge = (edge[0], u)
            if edge[0] != edge[1]:
                new_edges.append(edge)
        self.edges = new_edges
                
        
    def contract_algorithm(self, t=2):
        while self.V > t:
            u, v = random.choice(self.edges)
            self.contract_edge(u, v)
        return self.E
    
    def brute_force_cut(self):
        """try all possible cuts and return the minimum one"""
        
        min_cut = float('inf')
        vertices = list(set(itertools.chain.from_iterable(self.edges)))
        n = len(vertices)
        
        for i in range(1, int(2**(n-1))):
            cut_set = set()
            for j in range(n):
                if i & (1 << j):
                    cut_set.add(vertices[j])
                
            cut_edges = 0
            for u, v in self.edges:
                if (u in cut_set and v not in cut_set) or (v in cut_set and u not in cut_set):
                    cut_edges += 1
            
            if cut_edges < min_cut:
                min_cut = cut_edges
        
        return min_cut
    
    #@lru_cache(None)
    def fast_cut_algorithm(self):
        
        if self.V <= 6:
            return self.brute_force_cut()
        
        H1 = copy.deepcopy(self)
        H2 = copy.deepcopy(self)
        
        t = math.ceil(1+ self.V / math.sqrt(2))
        
        H1.contract_algorithm(t)
        H2.contract_algorithm(t)
        
        H1_min_cut = H1.fast_cut_algorithm()
        H2_min_cut = H2.fast_cut_algorithm()
        
        return min(H1_min_cut, H2_min_cut)
    
    
    def create_random_graph(self, V=10, E=20):
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
            self.add_edge(u, v)
        self.remember_edges()
    
    def create_complete_graph(self, V):
        for u in range(V):
            for v in range(u + 1, V):
                self.add_edge(u, v)
        self.remember_edges
    
    def create_cycle_graph(self, V):
        for u in range(V):
            self.add_edge(u, (u + 1) % V)
        self.remember_edges()
    
    def create_star_graph(self, V):
        for u in range(1, V):
            self.add_edge(0, u)
        self.remember_edges()
    
        
    def create_graph_png(self, filename):
        G = pgv.AGraph(strict=False, directed=False)
        for edge in self.edges:
            G.add_edge(*edge)
        G.draw(filename, prog="dot", format="png")