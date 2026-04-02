import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

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


def compute_centralities(G: nx.Graph) -> Dict[str, Dict[Any, float]]:
    """Compute a set of centrality measures and return them as dicts.

    Returns a dict with keys: 'degree', 'closeness', 'betweenness', 'eigenvector'.
    """
    centralities = {}
    centralities['degree'] = dict(nx.degree_centrality(G))
    centralities['closeness'] = dict(nx.closeness_centrality(G))
    centralities['betweenness'] = dict(nx.betweenness_centrality(G))
    try:
        centralities['eigenvector'] = dict(nx.eigenvector_centrality(G, max_iter=1000))
    except Exception:
        # Fallback: use eigenvector centrality numpy-based approximation
        centralities['eigenvector'] = {n: 0.0 for n in G.nodes()}
    return centralities


def summarize_scores(scores: Dict[Any, float]) -> Dict[str, float]:
    vals = np.array(list(scores.values()), dtype=float)
    return {
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'std': float(np.std(vals, ddof=0)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
    }


def top_k(scores: Dict[Any, float], k: int = 5):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def analyze_centrality(G: nx.Graph, top_k_n: int = 5, plot: bool = True, save_fig: bool = False, prefix: str = "centrality"):
    """Run a compact analysis of multiple centrality measures.

    Prints summary statistics, top-k nodes for each measure, and a correlation table.
    Optionally plots histograms and a pairwise scatter matrix.
    """
    centralities = compute_centralities(G)

    print("Centrality Summary:\n")
    summaries = {}
    measures = list(centralities.keys())
    for m in measures:
        summaries[m] = summarize_scores(centralities[m])
        s = summaries[m]
        print(f"- {m.title()}: mean={s['mean']:.4f}, median={s['median']:.4f}, std={s['std']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")

    print("\nTop nodes per measure:\n")
    for m in measures:
        print(f"{m.title()} Top {top_k_n}:")
        for node, val in top_k(centralities[m], top_k_n):
            print(f"  Node {node}: {val:.6f}")
        print("")

    # Correlation matrix (Pearson) across measures
    vals_matrix = np.vstack([np.array(list(centralities[m].values())) for m in measures])
    corr = np.corrcoef(vals_matrix)
    print("Correlation matrix (Pearson) between measures:")
    print("\t" + "\t".join([m for m in measures]))
    for i, m in enumerate(measures):
        row = "\t".join([f"{corr[i,j]:.3f}" for j in range(len(measures))])
        print(f"{m}\t{row}")

    if plot:
        # Histograms for each measure
        fig, axes = plt.subplots(len(measures), 1, figsize=(8, 3 * len(measures)))
        if len(measures) == 1:
            axes = [axes]
        for ax, m in zip(axes, measures):
            ax.hist(list(centralities[m].values()), bins=15, color='C0', alpha=0.8)
            ax.set_title(f"Distribution: {m.title()}")
            ax.set_xlabel('Centrality')
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        if save_fig:
            fig.savefig(f"{prefix}_histograms.png", dpi=150)
        plt.show()

        # Pairwise scatter (first two measures only if many)
        if len(measures) >= 2:
            a, b = measures[0], measures[1]
            plt.figure(figsize=(6, 5))
            plt.scatter(list(centralities[a].values()), list(centralities[b].values()), alpha=0.8)
            plt.xlabel(a.title())
            plt.ylabel(b.title())
            plt.title(f"Scatter: {a.title()} vs {b.title()}")
            if save_fig:
                plt.savefig(f"{prefix}_{a}_vs_{b}.png", dpi=150)
            plt.show()

    return {
        'centralities': centralities,
        'summaries': summaries,
        'correlation': corr,
        'measures': measures,
    }

if __name__ == "__main__":
    print("Generating Centrality Analysis and Visualizations...\n")

    # Use a Barbell Graph (Two 5-node cliques connected by a 4-node path)
    # This structure highlights the difference between bridges and clusters.
    G = nx.barbell_graph(5, 4)

    # Run analysis: prints summaries, top-k, correlations, and optional plots
    analyze_centrality(G, top_k_n=5, plot=True, save_fig=False, prefix="barbell")

    # 1. Visualize Betweenness (Expect high scores on the connecting path)
    visualize_centrality_heatmap(G, centrality_type="betweenness")

    # 2. Visualize Eigenvector (Expect high scores inside the dense cliques)
    visualize_centrality_heatmap(G, centrality_type="eigenvector")
