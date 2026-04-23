import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def algebraic_connectivity(G: nx.Graph) -> float:
    """Return lambda_2 of the normalized Laplacian (Fiedler value)."""
    if G.number_of_nodes() < 2:
        return 0.0
    if not nx.is_connected(G):
        # For disconnected graphs, lambda_2 is exactly 0.
        return 0.0

    # The midterm datasets are small enough for a deterministic dense solve,
    # which avoids iterative-solver jitter between repeated removals.
    if G.number_of_nodes() <= 250:
        laplacian = nx.normalized_laplacian_matrix(G).astype(float).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    # Use NetworkX's built in algebraic connectivity for much faster execution on larger graphs
    try:
        return float(nx.algebraic_connectivity(G, tol=1e-4, method='lanczos', seed=42, normalized=True))
    except Exception:
        # Fallback to dense symmetric eigensolver for numerical stability.
        laplacian = nx.normalized_laplacian_matrix(G).astype(float).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def compute_centralities(G: nx.Graph) -> Dict[str, Dict[Any, float]]:
    """Compute centrality scores used in the stability analysis."""
    centralities: Dict[str, Dict[Any, float]] = {}
    centralities["degree"] = dict(nx.degree_centrality(G))
    centralities["closeness"] = dict(nx.closeness_centrality(G))
    centralities["betweenness"] = dict(nx.betweenness_centrality(G))

    try:
        centralities["eigenvector"] = dict(nx.eigenvector_centrality(G, max_iter=1000))
    except Exception:
        centralities["eigenvector"] = {n: 0.0 for n in G.nodes()}

    return centralities


def normalize_scores(scores: Dict[Any, float]) -> Dict[Any, float]:
    vals = np.array(list(scores.values()), dtype=float)
    low = float(np.min(vals))
    high = float(np.max(vals))

    if np.isclose(high, low):
        return {k: 0.0 for k in scores}

    return {k: (v - low) / (high - low) for k, v in scores.items()}


def removal_impact_scores(G: nx.Graph, baseline_lambda2: float) -> Dict[Any, float]:
    """For each node, compute lambda_2 drop after removing that node."""
    impacts: Dict[Any, float] = {}

    for node in G.nodes():
        H = G.copy()
        H.remove_node(node)
        new_lambda2 = algebraic_connectivity(H)
        impacts[node] = max(0.0, baseline_lambda2 - new_lambda2)

    return impacts


def combine_criticality(
    betweenness: Dict[Any, float],
    eigenvector: Dict[Any, float],
    removal_impact: Dict[Any, float],
) -> Dict[Any, float]:
    """Blend bridge-importance, hub-importance, and structural fragility into one score."""
    b = normalize_scores(betweenness)
    e = normalize_scores(eigenvector)
    r = normalize_scores(removal_impact)

    combined: Dict[Any, float] = {}
    for node in betweenness:
        combined[node] = 0.45 * b[node] + 0.25 * e[node] + 0.30 * r[node]

    return combined


def top_k(scores: Dict[Any, float], k: int = 5) -> List[Tuple[Any, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


def graph_stability_label(lambda2: float) -> str:
    if lambda2 < 0.10:
        return "Very Fragile"
    if lambda2 < 0.50:
        return "Moderately Fragile"
    if lambda2 < 1.50:
        return "Stable"
    return "Highly Stable"


def make_dataset(name: str, seed: int) -> nx.Graph:
    if name == "karate":
        return nx.karate_club_graph()
    if name == "barbell":
        return nx.barbell_graph(8, 4)
    if name == "erdos":
        return nx.erdos_renyi_graph(40, 0.10, seed=seed)

    raise ValueError(f"Unsupported dataset: {name}")


def visualize_critical_nodes(
    G: nx.Graph,
    criticality: Dict[Any, float],
    top_nodes: List[Any],
    output_path: Path,
    seed: int,
) -> None:
    pos = nx.spring_layout(G, seed=seed)
    node_colors = [criticality[n] for n in G.nodes()]
    node_sizes = [500 + 1600 * criticality[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.inferno,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=top_nodes,
        node_color=[criticality[n] for n in top_nodes],
        cmap=plt.cm.inferno,
        node_size=[500 + 1600 * criticality[n] for n in top_nodes],
        edgecolors="cyan",
        linewidths=2.5,
        ax=ax,
    )

    labels = {n: str(n) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)

    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("Combined Criticality Score")

    ax.set_title("Connectivity Analyzer: Critical Node Visualization")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_report(
    G: nx.Graph,
    dataset: str,
    lambda2: float,
    label: str,
    centralities: Dict[str, Dict[Any, float]],
    removal_impact: Dict[Any, float],
    criticality: Dict[Any, float],
    top_n: int,
    figure_path: Path,
) -> Dict[str, Any]:
    top_critical = top_k(criticality, top_n)

    report = {
        "dataset": dataset,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G),
        "algebraic_connectivity_lambda2": lambda2,
        "stability_label": label,
        "top_critical_nodes": [
            {
                "node": int(node) if isinstance(node, (int, np.integer)) else node,
                "criticality": float(score),
                "betweenness": float(centralities["betweenness"][node]),
                "eigenvector": float(centralities["eigenvector"][node]),
                "lambda2_drop_after_removal": float(removal_impact[node]),
            }
            for node, score in top_critical
        ],
        "critical_node_visualization": str(figure_path),
    }
    return report


def write_text_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("Connectivity Analyzer Report")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"Dataset: {report['dataset']}")
    lines.append(f"Nodes: {report['nodes']}")
    lines.append(f"Edges: {report['edges']}")
    lines.append(f"Connected Components: {report['connected_components']}")
    lines.append(f"Algebraic Connectivity (lambda_2): {report['algebraic_connectivity_lambda2']:.6f}")
    lines.append(f"Stability Label: {report['stability_label']}")
    lines.append("")
    lines.append("Top Critical Nodes:")

    for rank, row in enumerate(report["top_critical_nodes"], start=1):
        lines.append(
            f"{rank}. Node {row['node']} | Combined={row['criticality']:.6f} | "
            f"Betweenness={row['betweenness']:.6f} | "
            f"Eigenvector={row['eigenvector']:.6f} | "
            f"lambda_2 drop={row['lambda2_drop_after_removal']:.6f}"
        )

    lines.append("")
    lines.append(f"Critical Node Visualization: {report['critical_node_visualization']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Connectivity Analyzer")
    parser.add_argument("--dataset", choices=["karate", "barbell", "erdos"], default="karate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--report-json", default="outputs/stability_report.json")
    parser.add_argument("--report-txt", default="outputs/stability_report.txt")
    parser.add_argument("--plot-path", default="outputs/critical_nodes.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = make_dataset(args.dataset, args.seed)

    lambda2 = algebraic_connectivity(graph)
    label = graph_stability_label(lambda2)

    centralities = compute_centralities(graph)
    removal_impact = removal_impact_scores(graph, lambda2)
    criticality = combine_criticality(
        centralities["betweenness"],
        centralities["eigenvector"],
        removal_impact,
    )

    top_nodes = [node for node, _ in top_k(criticality, args.top_n)]
    plot_path = Path(args.plot_path)
    visualize_critical_nodes(graph, criticality, top_nodes, plot_path, args.seed)

    report = build_report(
        graph,
        args.dataset,
        lambda2,
        label,
        centralities,
        removal_impact,
        criticality,
        args.top_n,
        plot_path,
    )

    json_path = Path(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_path = Path(args.report_txt)
    write_text_report(report, txt_path)

    print("Connectivity Analyzer completed.")
    print(f"- Stability label: {label}")
    print(f"- Algebraic connectivity (lambda_2): {lambda2:.6f}")
    print(f"- JSON report: {json_path}")
    print(f"- Text report: {txt_path}")
    print(f"- Visualization: {plot_path}")


if __name__ == "__main__":
    main()
