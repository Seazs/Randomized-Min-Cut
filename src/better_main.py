import random
import math
from tqdm import tqdm
from graph import Graph
import networkx as nx
import time
import matplotlib.pyplot as plt



def deterministic_min_cut(graph):
    """
    use networkx to find the min cut
    """
    G = nx.Graph()
    G.add_edges_from(graph.edges)
    return nx.minimum_edge_cut(G)


def find_min_cut_with_contract(graph, num_trials):
    min_cut = len(graph.edges)
    for _ in range(num_trials):
        graph.contract_algorithm()
        if len(graph.edges) < min_cut:
            min_cut = len(graph.edges)
        graph.reset_graph()
    return min_cut

def find_min_cut_with_fast_cut(graph, num_trials):
    min_cut = len(graph.edges)
    for _ in range(num_trials):
        graph.fast_cut_algorithm()
        if len(graph.edges) < min_cut:
            min_cut = len(graph.edges)
        graph.reset_graph()
    return min_cut

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
            cut = graph.contract_algorithm()
        else:
            cut = graph.fast_cut_algorithm()
        if cut == min_cut:
            success += 1
        graph.reset_graph()
    return success / n_test



    

def plot_results(sizes, contract_times, fast_cut_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [contract_time*100000 for contract_time in contract_times], label='Contract Algorithm Time', marker='o')
    plt.plot(sizes, [fast_cut_time*1000 for fast_cut_time in fast_cut_times], label='Fast Cut Algorithm Time', marker='o')
    plt.plot(sizes, [(size**2) for size in sizes], label='n²', linestyle='--')
    plt.plot(sizes, [(size**2)*math.log(size) for size in sizes], label='n² log(n)', linestyle='--')
    plt.xlabel('Graph Size')
    plt.ylabel('Time (ms)')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_algorithms(graph):
    sizes = []
    contract_times = []
    fast_cut_times = []
    
    for size in range(10, 120, 10):
        m = (4/4)*(size*(size-1)/2) - 1
        graph = Graph(V=size, E=m)
        sizes.append(size)
        
        print(f"size: {size}")
        
        print("contract algo")
        start = time.time()
        graph.contract_algorithm()
        contract_times.append(time.time() - start)
        graph.reset_graph()
        
        print("fast cut algo")
        start = time.time()
        graph.fast_cut_algorithm()
        fast_cut_times.append(time.time() - start)
        graph.reset_graph()
    
    print(sizes)
    print(contract_times)
    print(fast_cut_times)
    plot_results(sizes, contract_times, fast_cut_times)
        
        







        
        

if __name__ == "__main__":
    
    n = 15
    m = (3/4)*(n*(n-1)/2) - 1
    num_trials = n * (n - 1) * int(math.log(n))
    #num_trials = 100
    n_test = 500

    print(f"n: {n}, m: {m}, num_trials: {num_trials}")

    graph = Graph("star", n)
    
    graph.create_graph_png("./output/initial_graph.png")
    
    min_cut = deterministic_min_cut(graph)
    print("real min cut : " + str(min_cut))
    print(len(min_cut))
    
    
    # print("theorical error probability for contract algo : " + str(compute_theorical_error_probability_contract(n)))
    # print("theorical error probability for fast cut algo : " + str(compute_theorical_error_probability_fast_cut(n)))
    
    # print("empirical success probability for contract algo : " + str(compute_empirical_success_probability(graph, n_test, len(min_cut), algo = "contract")))
    # print("empirical success probability for fast cut algo : " + str(compute_empirical_success_probability(graph, n_test, len(min_cut), algo = "fast_cut")))
        

    compare_algorithms(graph)