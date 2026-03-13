import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Erdős-Rényi Graph (Random Graph)
def erdos_renyi_demo():
    n = 20  # Number of nodes
    p = 0.2  # Probability of edge creation
    G = nx.erdos_renyi_graph(n, p)
    plt.figure(figsize=(6, 4))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
    plt.title("Erdős-Rényi Graph (n=20, p=0.2)")
    plt.show()

# Barabási-Albert Graph (Scale-Free Graph)
def barabasi_albert_demo():
    n = 20  # Number of nodes
    m = 2   # Number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(n, m)
    plt.figure(figsize=(6, 4))
    nx.draw(G, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title("Barabási-Albert Graph (n=20, m=2)")
    plt.show()

print("Erdős-Rényi Graph Demo:")
erdos_renyi_demo()
print("Barabási-Albert Graph Demo:")
barabasi_albert_demo()