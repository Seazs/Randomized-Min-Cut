#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>


// Structure to represent an edge
struct Edge {
    int u, v; // Vertices connected by the edge
};

// Class to represent a graph
class Graph {
private:
    int V, E; // Number of vertices and edges
    std::vector<Edge> edges; // Array of edges

public:
    Graph() : V(0), E(0) {
        // seed
        std::srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    }

    // Get the list of edges
    std::vector<Edge> getEdges() {
        return edges;
    }

    // Add an edge to the graph
    void addEdge(Edge edge) {
        edges.push_back(edge);
        this->E = edges.size();
        this->V = std::max({this->V, edge.u + 1, edge.v + 1});
    }

    // Print the graph
    void printGraph() {
        for (int i = 0; i < E; i++) {
            std::cout << edges[i].u << " -- " << edges[i].v << std::endl;
        }
    }

    // Contract an edge (u, v) in the graph
    void contractEdge(int u, int v) {
        for (auto &edge : this->edges) {
            if (edge.u == v) edge.u = u;
            if (edge.v == v) edge.v = u;
        }

        // Remove self-loops
        this->edges.erase(std::remove_if(this->edges.begin(), this->edges.end(), [u](Edge &edge) {
            return edge.u == edge.v;
        }), this->edges.end());

        // Update the number of vertices and edges
        this->V--;
        this->E = this->edges.size();
    }

    // Karger's algorithm to find the minimum cut
    std::vector<Edge> contract_algorithm() {
        std::vector<Edge> edges_copy = this->edges;
        int original_V = this->V;
        while (this->V > 2) {
            // Pick a random edge
            int random_edge = rand() % this->E;
            int u = edges_copy[random_edge].u;
            int v = edges_copy[random_edge].v;

            // Contract the edge
            this->contractEdge(u, v);
        }
        this->V = original_V; // Reset the number of vertices
        std::vector<Edge> contracted_edges = this->edges;
        this->edges = edges_copy; // Reset the edges
        this->E = this->edges.size(); // Reset the number of edges
        return contracted_edges;
    }

    

    // Create a PNG image of the graph using Graphviz
    void create_graph_png(const std::string &filename) {
        std::ofstream file("graph.dot");
        file << "graph G {\n";
        for (auto &edge : this->edges) {
            file << edge.u << " -- " << edge.v << ";\n";
        }
        file << "}";
        file.close();
        system(("dot -Tpng graph.dot -o " + filename).c_str());
    }
};

// Create a random graph with V vertices and E edges
Graph create_random_graph(int V, int E) {
    Graph graph;
    std::set<std::pair<int, int>> unique_edges;
    while (unique_edges.size() < E) {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v) {
            auto edge = std::minmax(u, v);
            unique_edges.insert(edge);
        }
    }
    for (const auto &edge : unique_edges) {
        graph.addEdge({edge.first, edge.second});
    }
    return graph;
}

// Find the minimum cut of the graph using Karger's algorithm
int find_min_cut(Graph graph, int num_trials) {
    int min_cut = graph.getEdges().size(); // Initialize the minimum cut to the maximum possible value
    for (int i = 0; i < num_trials; i++) {
        std::vector<Edge> contracted_edges = graph.contract_algorithm();
        std::cout << "contracted_edges.size(): " << contracted_edges.size() << std::endl;
        min_cut = std::min(min_cut, (int) contracted_edges.size());
    }
    return min_cut;
}

// Benchmark the algorithm by running it multiple times
float benchmark(Graph graph, int num_trials, int min_cut) {
    int min_cut_count = 0;
    for (int i = 0; i < num_trials; i++) {
        std::vector<Edge> contracted_edges = graph.contract_algorithm();
        if (contracted_edges.size() == min_cut) {
            min_cut_count++;
        }
    }
    return (float) min_cut_count / num_trials;
}

int main() {

    int n = 10;
    int m = n * (n - 1) / 2 - 1;

    // int num_trials = (int) n*(n-1) * log(n);
    int num_trials = 10000;

    // Create a graph with 5 vertices and 5 edges
    Graph graph = create_random_graph(n, m);

    // Print the graph
    graph.create_graph_png("initial_graph.png");

    // Find the minimum cut
    int min_cut = find_min_cut(graph, num_trials);
    std::cout << "Minimum cut: " << min_cut << std::endl;

    // Benchmark the algorithm
    float success_rate = benchmark(graph, num_trials, min_cut);
    std::cout << "Success rate: " << success_rate << std::endl;



    return 0;
}