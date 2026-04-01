import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def visualize_centrality_heatmap(G, centrality_type="betweenness"):
    """
    Calculates node centrality and creates a heatmap visualization.
    The node color corresponds to its relative centrality score.
    """
    # Calculate centrality scores
    if centrality_type == "betweenness":
        scores = nx.betweenness_centrality(G)
        title = "Heatmap: Betweenness Centrality"
    elif centrality_type == "eigenvector":
        # Uses the power iteration method to find the principal eigenvector
        scores = nx.eigenvector_centrality(G, max_iter=1000)
        title = "Heatmap: Eigenvector Centrality"
    else:
        raise ValueError("Unsupported centrality type.")

    # Convert scores to a list for color mapping
    node_colors = [scores[node] for node in G.nodes()]
    
    # Position nodes using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 7))
    
    # Draw the graph with a heatmap colormap
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        cmap=plt.cm.plasma, 
        node_size=600,
        edgecolors='black'
    )
    
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")
    
    # Add a colorbar to interpret the heatmap
    plt.colorbar(nodes, label="Centrality Score")
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    print("Generating Centrality Visualizations...\n")

    # Use a Barbell Graph (Two 5-node cliques connected by a 4-node path)
    # This structure highlights the difference between bridges and clusters.
    G = nx.barbell_graph(5, 4)

    # 1. Visualize Betweenness (Expect high scores on the connecting path)
    visualize_centrality_heatmap(G, centrality_type="betweenness")

    # 2. Visualize Eigenvector (Expect high scores inside the dense cliques)
    visualize_centrality_heatmap(G, centrality_type="eigenvector")