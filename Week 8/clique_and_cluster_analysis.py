import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


SEED = 42
MIN_CLIQUE_SIZE = 3
CLIQUE_OVERLAP_THRESHOLD = 0.5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def create_graph(n, m, graph_type="ba", cluster_size=10):
    """
    Creates a graph for the Week 8 experiments.

    Kept compatible with the old create_graph(n, m) interface:
      - m == "complete" makes a complete graph
      - otherwise it makes a Barabasi-Albert graph

    Extended graph_type values:
      - "ba": Barabasi-Albert graph
      - "complete": complete graph
      - "planted": connected caveman graph with obvious planted clusters
    """
    if m == "complete" or graph_type == "complete":
        print(f"Generating Complete graph (n={n})...")
        return nx.complete_graph(n)

    if graph_type == "planted":
        clique_count = max(2, int(np.ceil(n / cluster_size)))
        print(
            "Generating planted-cluster graph "
            f"({clique_count} clusters, cluster_size={cluster_size})..."
        )
        G = nx.connected_caveman_graph(clique_count, cluster_size)
        if G.number_of_nodes() > n:
            G = G.subgraph(range(n)).copy()
        return nx.convert_node_labels_to_integers(G)

    print(f"Generating Barabasi-Albert graph (n={n}, m={m})...")
    return nx.barabasi_albert_graph(n=n, m=m, seed=SEED)


def graph_is_complete(G):
    n = G.number_of_nodes()
    return n <= 1 or G.number_of_edges() == n * (n - 1) // 2


def jaccard_overlap(a, b):
    union_size = len(a | b)
    if union_size == 0:
        return 0.0
    return len(a & b) / union_size


def internal_density(G, cluster):
    nodes = set(cluster)
    size = len(nodes)
    if size <= 1:
        return 0.0
    possible_edges = size * (size - 1) / 2
    return G.subgraph(nodes).number_of_edges() / possible_edges


def cut_edges_count(G, cluster):
    nodes = set(cluster)
    return sum(1 for u, v in G.edges() if (u in nodes) != (v in nodes))


def merge_overlapping_clusters(clusters, overlap_threshold=CLIQUE_OVERLAP_THRESHOLD):
    merged = [set(cluster) for cluster in clusters]
    changed = True

    while changed:
        changed = False
        next_clusters = []
        used = [False] * len(merged)

        for i, cluster in enumerate(merged):
            if used[i]:
                continue

            current = set(cluster)
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if jaccard_overlap(current, merged[j]) >= overlap_threshold:
                    current |= merged[j]
                    used[j] = True
                    changed = True

            next_clusters.append(current)

        merged = next_clusters

    return merged


def assign_nodes_to_best_cluster(G, clusters):
    """
    Converts overlapping candidate clusters into non-overlapping clusters.
    A node in multiple candidates is assigned to the largest, densest cluster.
    """
    candidate_scores = [
        (len(cluster), internal_density(G, cluster), -idx)
        for idx, cluster in enumerate(clusters)
    ]
    node_to_cluster = {}

    for idx, cluster in enumerate(clusters):
        for node in cluster:
            current_idx = node_to_cluster.get(node)
            if current_idx is None or candidate_scores[idx] > candidate_scores[current_idx]:
                node_to_cluster[node] = idx

    assigned = {idx: set() for idx in range(len(clusters))}
    for node, idx in node_to_cluster.items():
        assigned[idx].add(node)

    return [cluster for cluster in assigned.values() if cluster]


def detect_clique_clusters(G, k_min=MIN_CLIQUE_SIZE, overlap_threshold=CLIQUE_OVERLAP_THRESHOLD):
    print(f"\nFinding clique clusters with cliques of at least {k_min} nodes...")

    if G.number_of_nodes() == 0:
        return []

    if graph_is_complete(G):
        print("Graph is complete; treating the whole graph as one clique cluster.")
        return [set(G.nodes())]

    maximal_cliques = [set(clique) for clique in nx.find_cliques(G)]
    large_cliques = [clique for clique in maximal_cliques if len(clique) >= k_min]
    print(f"Found {len(large_cliques)} maximal cliques with at least {k_min} nodes.")

    if not large_cliques:
        return []

    merged = merge_overlapping_clusters(large_cliques, overlap_threshold)
    assigned = [
        cluster
        for cluster in assign_nodes_to_best_cluster(G, merged)
        if len(cluster) >= k_min
    ]
    print(f"Merged into {len(assigned)} non-overlapping clique clusters.")
    return sorted(assigned, key=len, reverse=True)


def detect_communities(G):
    print("\nFinding communities using greedy modularity...")

    if G.number_of_nodes() == 0:
        return []

    if graph_is_complete(G):
        print("Graph is complete; treating the whole graph as one community.")
        return [set(G.nodes())]

    try:
        communities = [set(c) for c in nx.community.greedy_modularity_communities(G)]
    except Exception as exc:
        print(f"Community detection failed ({exc}); using the top hub neighborhood.")
        degrees = dict(G.degree())
        hub = max(degrees, key=degrees.get)
        communities = [set([hub] + list(G.neighbors(hub)))]

    print(f"Found {len(communities)} communities.")
    return sorted(communities, key=len, reverse=True)


def score_clusters(G, clusters):
    scores = []
    for idx, cluster in enumerate(clusters, start=1):
        cluster_set = set(cluster)
        scores.append(
            {
                "id": idx,
                "nodes": cluster_set,
                "size": len(cluster_set),
                "internal_edges": G.subgraph(cluster_set).number_of_edges(),
                "density": internal_density(G, cluster_set),
                "cut_edges": cut_edges_count(G, cluster_set),
            }
        )
    return scores


def print_cluster_report(G, clusters, label):
    print(f"\n{label} report:")
    if not clusters:
        print("  No clusters found.")
        return

    covered_nodes = set().union(*clusters)
    is_partition = covered_nodes == set(G.nodes()) and sum(len(c) for c in clusters) == len(covered_nodes)

    if len(clusters) > 1 and is_partition:
        modularity = nx.community.modularity(G, clusters)
        print(f"  Modularity: {modularity:.4f}")
    elif len(clusters) > 1:
        missing = G.number_of_nodes() - len(covered_nodes)
        print(f"  Modularity: skipped (partial cluster cover; {missing} nodes unassigned)")
    else:
        print("  Modularity: 0.0000 (single cluster)")

    for score in score_clusters(G, clusters):
        print(
            f"  Cluster {score['id']:>2}: size={score['size']:>3}, "
            f"density={score['density']:.3f}, cut_edges={score['cut_edges']:>3}, "
            f"internal_edges={score['internal_edges']:>3}"
        )


def largest_component_size(G):
    if G.number_of_nodes() == 0:
        return 0
    return max(len(component) for component in nx.connected_components(G))


def cluster_attack_test(G, clusters, strategy="bridge_edges"):
    print(f"\nRunning cluster attack test with strategy='{strategy}'...")
    if not clusters:
        print("No clusters available for cluster attack.")
        return []

    scores = score_clusters(G, clusters)
    if strategy == "bridge_edges":
        ordered = sorted(scores, key=lambda item: item["cut_edges"], reverse=True)
    elif strategy == "size":
        ordered = sorted(scores, key=lambda item: item["size"], reverse=True)
    else:
        raise ValueError(f"Unknown cluster attack strategy: {strategy}")

    G_test = G.copy()
    results = [
        {
            "step": 0,
            "removed": None,
            "removed_count": 0,
            "largest_component_size": largest_component_size(G_test),
        }
    ]

    for step, score in enumerate(ordered, start=1):
        nodes_to_remove = [node for node in score["nodes"] if node in G_test]
        G_test.remove_nodes_from(nodes_to_remove)
        component_size = largest_component_size(G_test)
        results.append(
            {
                "step": step,
                "removed": score["id"],
                "removed_count": len(nodes_to_remove),
                "largest_component_size": component_size,
            }
        )
        print(
            f"  Step {step}: removed cluster {score['id']} "
            f"({len(nodes_to_remove)} nodes, cut_edges={score['cut_edges']}) | "
            f"largest component={component_size}"
        )

    return results


def hub_attack_test(G, max_steps):
    print("\nRunning single-node hub attack baseline...")
    G_test = G.copy()
    results = [
        {
            "step": 0,
            "removed": None,
            "removed_count": 0,
            "largest_component_size": largest_component_size(G_test),
        }
    ]

    for step in range(1, max_steps + 1):
        if G_test.number_of_nodes() == 0:
            break
        degrees = dict(G_test.degree())
        hub = max(degrees, key=degrees.get)
        G_test.remove_node(hub)
        results.append(
            {
                "step": step,
                "removed": hub,
                "removed_count": 1,
                "largest_component_size": largest_component_size(G_test),
            }
        )

    return results


def color_nodes_by_clusters(G, clusters):
    tab20 = plt.colormaps.get_cmap("tab20")
    node_to_color = {node: "lightgray" for node in G.nodes()}

    for idx, cluster in enumerate(clusters):
        color = tab20(idx % 20)
        for node in cluster:
            if node in node_to_color:
                node_to_color[node] = color

    return [node_to_color[node] for node in G.nodes()]


def visualize_clusters(G, clusters, pos, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    node_colors = color_nodes_by_clusters(G, clusters)
    node_labels = {node: node for node in G.nodes()}
    degrees = dict(G.degree())
    node_sizes = [80 + degrees[node] * 18 for node in G.nodes()]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.22, edge_color="gray")
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)

    summary = ", ".join(str(len(cluster)) for cluster in clusters[:6])
    if len(clusters) > 6:
        summary += ", ..."
    plt.title(
        f"{title}\n{len(clusters)} clusters | sizes: {summary or 'none'}",
        fontsize=15,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved cluster visualization to {output_path}")
    plt.close()


def plot_attack_comparison(cluster_results, hub_results, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    cluster_steps = [item["step"] for item in cluster_results]
    cluster_components = [item["largest_component_size"] for item in cluster_results]
    hub_steps = [item["step"] for item in hub_results]
    hub_components = [item["largest_component_size"] for item in hub_results]

    plt.plot(
        cluster_steps,
        cluster_components,
        marker="o",
        linewidth=2,
        label="Cluster removal",
    )
    plt.plot(
        hub_steps,
        hub_components,
        marker="s",
        linewidth=2,
        label="Single-node hub removal",
    )
    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel("Attack step")
    plt.ylabel("Largest connected component size")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved attack comparison to {output_path}")
    plt.close()


def run_graph_analysis(config):
    graph_type = config["type"]
    n = config["n"]
    m = config["m"]

    G = create_graph(
        n=n,
        m=m,
        graph_type=graph_type,
        cluster_size=config.get("cluster_size", 10),
    )
    pos = nx.spring_layout(G, seed=SEED, k=config.get("layout_k"))
    label = f"{graph_type}_n{n}_m{m}"

    clique_clusters = detect_clique_clusters(
        G,
        k_min=config.get("k_min", MIN_CLIQUE_SIZE),
        overlap_threshold=CLIQUE_OVERLAP_THRESHOLD,
    )
    print_cluster_report(G, clique_clusters, "Clique cluster")
    visualize_clusters(
        G,
        clique_clusters,
        pos,
        f"{graph_type.upper()} graph: clique-aware clusters",
        os.path.join(OUTPUT_DIR, "cliques", f"{label}_clique_clusters.png"),
    )

    communities = detect_communities(G)
    print_cluster_report(G, communities, "Community")
    visualize_clusters(
        G,
        communities,
        pos,
        f"{graph_type.upper()} graph: greedy modularity communities",
        os.path.join(OUTPUT_DIR, "communities", f"{label}_communities.png"),
    )

    attack_clusters = communities if communities else clique_clusters
    cluster_results = cluster_attack_test(G, attack_clusters, strategy="bridge_edges")
    hub_results = hub_attack_test(G, max_steps=max(1, len(cluster_results) - 1))
    plot_attack_comparison(
        cluster_results,
        hub_results,
        f"{graph_type.upper()} graph: cluster vs hub attack",
        os.path.join(OUTPUT_DIR, "attacks", f"{label}_attack_comparison.png"),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    graph_configs = [
        {"n": 50, "m": 2, "type": "ba", "k_min": 3, "layout_k": 0.25},
        {"n": 30, "m": "complete", "type": "complete", "k_min": 3},
        {"n": 48, "m": "clustered", "type": "planted", "cluster_size": 8, "k_min": 4},
    ]

    for config in graph_configs:
        print("\n" + "=" * 72)
        run_graph_analysis(config)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
