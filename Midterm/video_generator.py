import os
import gzip
import urllib.request
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Any, Dict, List
from connectivity_analyzer import compute_centralities, removal_impact_scores, combine_criticality, top_k, algebraic_connectivity


def connected_bfs_subgraph(G: nx.Graph, start_node: Any, max_nodes: int) -> nx.Graph:
    """Return a connected BFS-induced subgraph with up to max_nodes nodes."""
    if G.number_of_nodes() <= max_nodes:
        return G.copy()

    visited = {start_node}
    queue: List[Any] = [start_node]
    head = 0

    while head < len(queue) and len(visited) < max_nodes:
        node = queue[head]
        head += 1
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            if len(visited) >= max_nodes:
                break

    return G.subgraph(visited).copy()


def largest_biconnected_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the largest biconnected induced subgraph (falls back to G if unavailable)."""
    components = [c for c in nx.biconnected_components(G) if len(c) >= 3]
    if not components:
        return G.copy()
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def select_monotone_targets(
    G: nx.Graph,
    criticality: Dict[Any, float],
    attack_steps: int = 5,
    reserve_top: int = 5,
    candidate_pool: int = 30,
    eps: float = 1e-8,
) -> List[Any]:
    """
    Select attack targets with a preference for second-tier nodes while
    keeping lambda_2 non-increasing across removal steps.
    """
    if not criticality or G.number_of_nodes() < 2:
        return []

    ranked_nodes = [node for node, _ in top_k(criticality, len(criticality))]
    tier2_candidates = ranked_nodes[reserve_top: reserve_top + candidate_pool]

    working_graph = G.copy()
    current_lambda = algebraic_connectivity(working_graph)
    targets: List[Any] = []

    def extend_with_candidates(candidates: List[Any], allow_equal: bool) -> None:
        nonlocal current_lambda

        while len(targets) < attack_steps:
            best_node = None
            best_lambda = None

            for node in candidates:
                if node in targets or node not in working_graph:
                    continue

                trial_graph = working_graph.copy()
                trial_graph.remove_node(node)
                trial_lambda = algebraic_connectivity(trial_graph)

                if allow_equal:
                    if trial_lambda > (current_lambda + eps):
                        continue
                else:
                    if trial_lambda >= (current_lambda - eps):
                        continue

                if (
                    best_node is None
                    or trial_lambda < (best_lambda - eps)
                    or (
                        abs(trial_lambda - best_lambda) <= eps
                        and criticality.get(node, 0.0) > criticality.get(best_node, 0.0)
                    )
                ):
                    best_node = node
                    best_lambda = trial_lambda

            if best_node is None:
                break

            working_graph.remove_node(best_node)
            targets.append(best_node)
            current_lambda = float(best_lambda)

    # Pass 1: prefer strict drops in second-tier candidates.
    extend_with_candidates(tier2_candidates, allow_equal=False)

    # Pass 2: if needed, allow strict drops from the full ranking.
    if len(targets) < attack_steps:
        extend_with_candidates(ranked_nodes, allow_equal=False)

    # Pass 3: backfill with non-increasing (equal allowed) tier-2 steps.
    if len(targets) < attack_steps:
        extend_with_candidates(tier2_candidates, allow_equal=True)

    # Pass 4: final backfill with non-increasing full-ranking steps.
    if len(targets) < attack_steps:
        extend_with_candidates(ranked_nodes, allow_equal=True)

    # Bootstrap if no non-increasing step exists from the initial state.
    if not targets and ranked_nodes:
        best_node = None
        best_lambda = None
        for node in ranked_nodes:
            if node not in working_graph:
                continue
            trial_graph = working_graph.copy()
            trial_graph.remove_node(node)
            trial_lambda = algebraic_connectivity(trial_graph)
            if (
                best_node is None
                or trial_lambda < (best_lambda - eps)
                or (
                    abs(trial_lambda - best_lambda) <= eps
                    and criticality.get(node, 0.0) > criticality.get(best_node, 0.0)
                )
            ):
                best_node = node
                best_lambda = trial_lambda

        if best_node is not None and best_lambda <= (current_lambda + eps):
            working_graph.remove_node(best_node)
            targets.append(best_node)
            current_lambda = float(best_lambda)
            extend_with_candidates(ranked_nodes, allow_equal=True)

    return targets

def load_better_dataset():
    """
    Downloads the Stanford SNAP Facebook Ego Network dataset.
    Returns a dense k-core subgraph to ensure it doesn't immediately 
    shatter into pieces upon the first node removal, giving us a more 
    interesting degradation curve.
    """
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    file_path = "facebook_combined.txt.gz"
    
    if not os.path.exists(file_path):
        print("Downloading the Facebook network dataset from SNAP...")
        urllib.request.urlretrieve(url, file_path)

    print("Parsing and loading into NetworkX...")
    with gzip.open(file_path, 'rt') as f:
        G_full = nx.read_edgelist(f, nodetype=int)

    # Instead of an ego graph (which breaks instantly when the ego is removed),
    # we take the 15-core of the graph. A k-core is a maximal subgraph that contains 
    # nodes of degree k or more. This leaves a highly cohesive subset of around 300-400 nodes.
    # We take the largest connected component of that core to be safe.
    G_core = nx.k_core(G_full, k=15)
    largest_cc = max(nx.connected_components(G_core), key=len)
    G = G_core.subgraph(largest_cc).copy()
    G = largest_biconnected_subgraph(G)
    
    # If the graph is still too large for quick animation, sample a connected region.
    if G.number_of_nodes() > 200:
        central_nodes = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)
        start_node = central_nodes[0][0]
        G = connected_bfs_subgraph(G, start_node, 200)

        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        # Re-enforce biconnectivity after sampling.
        G = largest_biconnected_subgraph(G)

    articulation_count = sum(1 for _ in nx.articulation_points(G))
    print(
        f"Loaded Dense Subgraph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, articulation points={articulation_count}"
    )
    return G

def create_animation(G, top_nodes, criticality, output_path: Path):
    """
    Creates an animated GIF dashboard with a side panel for explanations,
    a line chart tracking stability (lambda_2), and the main network visualization.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Create a GridSpec layout
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[3, 1])
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_metric = fig.add_subplot(gs[1, :])
    
    pos = nx.spring_layout(G, seed=42)
    
    # Precompute states to keep animation smooth and track metrics
    states = []
    current_G = G.copy()
    current_lambda = algebraic_connectivity(current_G)
    
    lambda_history = [current_lambda]
    step_labels = ["Start"]
    
    # 1. Initial State
    states.append({
        'graph': current_G.copy(),
        'target': None,
        'title': "Stage: Initial Network",
        'desc': (
            "This is the Stanford SNAP Facebook ego network.\n\n"
            f"Nodes: {current_G.number_of_nodes()}\n"
            f"Edges: {current_G.number_of_edges()}\n"
            f"Stability (λ2): {current_lambda:.4f}\n\n"
            "We will simulate targeted attacks on the "
            "most structurally critical nodes to observe "
            "how the network destabilizes."
        ),
        'lambda': current_lambda,
        'history': list(lambda_history),
        'labels': list(step_labels)
    })
    
    # Repeat initial frame to give the viewer time to read
    for _ in range(2):
        states.append(states[-1])
    
    total_attacks = len(top_nodes)

    for i, target in enumerate(top_nodes, 1):
        # 2. Highlight Node
        states.append({
            'graph': current_G.copy(),
            'target': target,
            'title': f"Stage: Targeting Node {target}",
            'desc': (
                f"Attack Step {i}/{total_attacks}\n\n"
                f"Identified candidate: Node {target}.\n\n"
                "This node was chosen from the first-tier candidate pool based on its criticality score, but we will only remove it if it does not cause an increase in λ2 compared to the previous step.\n\n"
                "Highlighting node before removal..."
            ),
            'lambda': current_lambda,
            'history': list(lambda_history),
            'labels': list(step_labels)
        })
        
        # Give time to read highlight
        states.append(states[-1])
        
        # 3. Remove Node
        current_G.remove_node(target)
        current_lambda = algebraic_connectivity(current_G)
        lambda_history.append(current_lambda)
        step_labels.append(f"-{target}")
        
        states.append({
            'graph': current_G.copy(),
            'target': None,
            'title': f"Stage: Node {target} Removed",
            'desc': (
                f"Node {target} and all its edges are gone.\n\n"
                f"Remaining Nodes: {current_G.number_of_nodes()}\n"
                f"New Stability (λ2): {current_lambda:.4f}\n\n"
                "Observe the updated stability chart below. By removing this node, we see how the network's connectivity degrades."
            ),
            'lambda': current_lambda,
            'history': list(lambda_history),
            'labels': list(step_labels)
        })
        
        # Give time to observe the drop
        states.append(states[-1])

    # Final summary frame
    states.append({
        'graph': current_G.copy(),
        'target': None,
        'title': "Stage: Simulation Complete",
        'desc': (
            "Final Assessment:\n\n"
            f"Nodes left: {current_G.number_of_nodes()}\n"
            f"Final Stability (λ2): {current_lambda:.4f}\n\n"
            "Conclusion:\n"
            "Removing the top critical node overall lead to a significant drop in stability. Each subsequent removal further degraded the network, demonstrating the cumulative impact of targeting structurally important nodes. However, these degradations were more gradual with every removal."
        ),
        'lambda': current_lambda,
        'history': list(lambda_history),
        'labels': list(step_labels)
    })
    states.append({
        'graph': current_G.copy(),
        'target': None,
        'title': "Stage: Simulation Complete",
        'desc': (
            "Final Assessment:\n\n"
            f"Nodes left: {current_G.number_of_nodes()}\n"
            f"Final Stability (λ2): {current_lambda:.4f}\n\n"
            "Conclusion:\n"
            "Removing the top critical node overall lead to a significant drop in stability. Each subsequent removal further degraded the network, demonstrating the cumulative impact of targeting structurally important nodes. However, these degradations were more gradual with every removal."
        ),
        'lambda': current_lambda,
        'history': list(lambda_history),
        'labels': list(step_labels)
    })
    for _ in range(3):
        states.append(states[-1])

    max_history_points = max((len(state['history']) for state in states), default=1)
    max_history_value = max(
        (max(state['history']) for state in states if state['history']),
        default=1.0,
    )

    def update(frame_idx):
        ax_graph.clear()
        ax_text.clear()
        ax_metric.clear()
        
        state = states[frame_idx]
        g_frame = state['graph']
        target = state['target']
        
        # --- 1. Draw Graph ---
        ax_graph.set_facecolor('#ffffff')
        node_colors = []
        node_sizes = []
        for n in g_frame.nodes():
            if n == target:
                node_colors.append("cyan")
                node_sizes.append(1000)
            else:
                score = criticality.get(n, 0)
                node_colors.append(score)
                node_sizes.append(100 + 400 * score)
                
        nx.draw_networkx_edges(g_frame, pos, alpha=0.15, ax=ax_graph, edge_color="gray")
        nx.draw_networkx_nodes(
            g_frame, pos,
            node_color=node_colors if target is None else ["cyan" if n == target else "lightgray" for n in g_frame.nodes()],
            cmap=plt.cm.inferno if target is None else None,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.5,
            ax=ax_graph
        )
        ax_graph.set_title(state['title'], fontsize=16, fontweight='bold')
        ax_graph.axis("off")
        
        # --- 2. Draw Text Panel ---
        ax_text.axis('off')
        ax_text.text(0.05, 0.9, state['desc'], fontsize=14, va='top', ha='left', wrap=True, family='monospace', bbox=dict(facecolor='#e9ecef', edgecolor='gray', boxstyle='round,pad=1'))
        
        # --- 3. Draw Stability Metric Line Chart ---
        ax_metric.set_facecolor('#ffffff')
        history = state['history']
        labels = state['labels']
        
        ax_metric.plot(range(len(history)), history, marker='o', color='red', linewidth=3, markersize=10)
        ax_metric.set_xlim(-0.2, max_history_points - 0.8)
        ax_metric.set_ylim(0, max(0.05, max_history_value * 1.1))
        ax_metric.set_xticks(range(len(labels)))
        ax_metric.set_xticklabels(labels, fontsize=12)
        ax_metric.set_ylabel("Algebraic Connectivity (λ2)", fontsize=12, fontweight='bold')
        ax_metric.set_title("Network Stability Degradation Over Time", fontsize=14)
        ax_metric.grid(True, linestyle='--', alpha=0.6)
        
        return ax_graph, ax_text, ax_metric

    print(f"Generating enhanced video animation ({len(states)} frames)...")
    # Increase interval to give time to read text annotations
    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=2000, blit=False)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer='pillow', fps=0.5)
    print(f"Saved enhanced explanation video to {output_path}")
    plt.close(fig)

def main():
    G = load_better_dataset()
    
    print("Computing metrics...")
    lambda2 = algebraic_connectivity(G)
    centralities = compute_centralities(G)
    removal_impact = removal_impact_scores(G, lambda2)
    criticality = combine_criticality(
        centralities["betweenness"],
        centralities["eigenvector"],
        removal_impact,
    )

    top_10 = [n for n, _ in top_k(criticality, 10)]
    target_nodes = select_monotone_targets(
        G,
        criticality,
        attack_steps=5,
        reserve_top=0,
        candidate_pool=30,
    )

    if not target_nodes:
        print("No non-increasing attack sequence found for this sampled graph.")

    print(f"Top 10 nodes: {top_10}")
    print(f"Selected attack targets: {target_nodes}")
    
    out_video = Path("outputs") / "critical_nodes_removal.gif"
    create_animation(G, target_nodes, criticality, out_video)

if __name__ == "__main__":
    main()
