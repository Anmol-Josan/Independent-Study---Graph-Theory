import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

def create_graph(n, m):
    """
    Creates a Barabasi-Albert graph or a complete graph.
    """
    if m == 'complete':
        print(f"Generating Complete graph (n={n})...")
        G = nx.complete_graph(n)
    else:
        print(f"Generating Barabasi-Albert graph (n={n}, m={m})...")
        G = nx.barabasi_albert_graph(n=n, m=m, seed=42)
    return G

def find_and_visualize_cliques(G, k_min=3, path_prefix="outputs/cliques"):
    """
    Finds cliques of a minimum size and visualizes them.
    """
    print(f"\nFinding cliques with at least {k_min} nodes...")
    cliques = list(nx.find_cliques(G))
    
    # Filter cliques by minimum size
    large_cliques = [c for c in cliques if len(c) >= k_min]
    print(f"Found {len(large_cliques)} cliques with at least {k_min} nodes.")

    if not large_cliques:
        print("No large cliques found to visualize.")
        return

    # Visualize the graph with cliques highlighted
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')

    node_colors = ['lightblue'] * G.number_of_nodes()
    node_labels = {node: node for node in G.nodes()}
    
    # Assign a unique color to each large clique
    tab20_colors = plt.colormaps.get_cmap('tab20')
    for i, clique in enumerate(large_cliques):
        color = tab20_colors(i % 20) # Cycle through tab20 colors
        for node in clique:
            node_colors[list(G.nodes()).index(node)] = color

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title(f"Graph with Detected Cliques (min size {k_min})", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    output_dir = os.path.dirname(path_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{path_prefix}_n{G.number_of_nodes()}.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved clique visualization to {filename}")
    plt.close()

def find_and_visualize_communities(G, path_prefix="outputs/communities"):
    """
    Finds communities using greedy modularity and visualizes them.
    """
    print("\nFinding communities using greedy modularity...")
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
    except Exception as e:
        print(f"Error finding communities: {e}. Falling back to a simpler approach.")
        # Fallback: treat top hub neighborhood as a community if modularity fails for small graphs
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            hub = max(degrees, key=degrees.get)
            communities = [set([hub] + list(G.neighbors(hub)))]
        else:
            print("Graph is empty, no communities to find.")
            return

    print(f"Found {len(communities)} communities.")
    
    if not communities:
        print("No communities found to visualize.")
        return

    # Visualize the graph with communities highlighted
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')

    node_colors = ['lightblue'] * G.number_of_nodes()
    node_labels = {node: node for node in G.nodes()}

    # Assign a unique color to each community
    tab20_colors = plt.colormaps.get_cmap('tab20')
    for i, community in enumerate(communities):
        color = tab20_colors(i % 20) # Cycle through tab20 colors
        for node in community:
            if node in G.nodes(): # Ensure node exists in graph
                node_colors[list(G.nodes()).index(node)] = color

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title("Graph with Detected Communities", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    output_dir = os.path.dirname(path_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{path_prefix}_n{G.number_of_nodes()}.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved community visualization to {filename}")
    plt.close()


if __name__ == "__main__":
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/cliques', exist_ok=True)
    os.makedirs('outputs/communities', exist_ok=True)

    # Example graph configurations
    graph_configs = [
        {"n": 50, "m": 2, "type": "ba"}, # Barabasi-Albert graph
        {"n": 30, "m": "complete", "type": "complete"} # Complete graph
    ]

    for config in graph_configs:
        n = config["n"]
        m = config["m"]
        graph_type = config["type"]

        G = create_graph(n, m)
        
        # Clique analysis
        find_and_visualize_cliques(G, k_min=3, path_prefix=f"outputs/cliques/{graph_type}_n{n}_m{m}_cliques")
        
        # Community analysis
        find_and_visualize_communities(G, path_prefix=f"outputs/communities/{graph_type}_n{n}_m{m}_communities")

    print("\nAnalysis complete.")
