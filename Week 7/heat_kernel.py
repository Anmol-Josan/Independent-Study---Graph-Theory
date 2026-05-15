import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os
import time
import random
#129

def plot_heat_traces(time_steps, heat_matrix, title, out_filename):
    plt.figure(figsize=(12, 7))

    # Plot transparent lines for all nodes
    for node_idx in range(heat_matrix.shape[1]):
        plt.plot(time_steps, heat_matrix[:, node_idx], alpha=0.3, color="gray")

    # Highlight the source node (the one that starts with max heat)
    source_idx = np.argmax(heat_matrix[0, :])
    plt.plot(
        time_steps,
        heat_matrix[:, source_idx],
        alpha=1.0,
        color="red",
        linewidth=2.5,
        label="Source Node",
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Time (t)", fontsize=14)
    plt.ylabel('Heat / "Disturbance" Level', fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_filename, dpi=150)
    print(f"Saved heat traces plot to {out_filename}")


def plot_graph_snapshots(
    G, pos, time_steps, heat_matrix, indices_to_plot, title, out_filename
):
    n_snapshots = len(indices_to_plot)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(6 * n_snapshots, 6))
    if n_snapshots == 1:
        axes = [axes]

    # Find global max heat for consistent color mapping
    max_heat = np.max(heat_matrix)

    for ax, idx in zip(axes, indices_to_plot):
        t = time_steps[idx]
        node_colors = heat_matrix[idx, :]

        # Use a colormap like "inferno", "hot" or "Reds" to simulate heat/disease
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=plt.cm.coolwarm,
            node_size=80,
            vmin=0,
            vmax=max_heat,
            edgecolors="black",
            linewidths=0.5,
        )
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")

        ax.set_title(f"t = {t:.1f}", fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    # Add a colorbar at the bottom
    cbar = fig.colorbar(
        nodes, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.5, pad=0.05
    )
    cbar.set_label("Heat Intensity", fontsize=12)

    plt.savefig(out_filename, dpi=150, bbox_inches="tight")
    print(f"Saved graph snapshots plot to {out_filename}")
    try:
        # plt.show()
        pass
    finally:
        plt.close()


def simulate_heat_equation(
    G, initial_node=None, initial_state=None, t_max=10.0, steps=100
):
    """
    Simulate heat diffusion on a graph G using the Heat Equation: H_t = e^{-tL}
    """
    n = G.number_of_nodes()
    # 1. Compute the Graph Laplacian L = D - A
    L = nx.laplacian_matrix(G).toarray()

    # 2. Define initial state x0
    x0 = np.zeros(n)
    nodes = list(G.nodes())
    if initial_state is not None:
        # Accept a full initial vector
        x0 = np.asarray(initial_state, dtype=float).reshape((n,))
        init_idx = None
    else:
        if initial_node is None:
            # Choose the node with the highest degree (a "hub") as the source
            degrees = dict(G.degree())
            initial_node = max(degrees, key=degrees.get)
        init_idx = nodes.index(initial_node)
        x0[init_idx] = 1.0  # Initial "heat" concentration or "disease" outbreak

    times = np.linspace(0, t_max, steps)
    heat_matrix = np.zeros((steps, n))

    # 3. Simulate over time
    if initial_state is not None:
        seeded = np.count_nonzero(x0)
        desc = f"initial vector with {seeded} seeded nodes"
    else:
        desc = f"node {initial_node} (degree {G.degree(initial_node)})"
    print(f"Simulating heat diffusion starting: {desc}...")
    for i, t in enumerate(times):
        # We calculate the heat kernel matrix: H_t = expm(-t * L)
        # In a very large graph, we'd use eigendecomposition or Chebyshev polynomials to approximate,
        # but for small/medium graphs, matrix exponential expm is fine.
        Ht = expm(-t * L)
        xt = Ht.dot(x0)
        heat_matrix[i, :] = xt

    return times, heat_matrix, init_idx


def make_initial_state(
    G, scenario="single_hub", k=3, fraction=0.05, seed=42, pos=None, radius=2
):
    """Create an initial heat vector for various scenarios.

    Scenarios supported:
      - 'single_hub' : one highest-degree node
      - 'top_k_hubs' : top-k highest-degree nodes equal weight
      - 'cluster_hub' : hub plus its immediate neighbors
      - 'random_k' : k random nodes
      - 'distributed' : random fraction of nodes with small heat
      - 'k_hop_area' : nodes within `radius` hops from the top hub
      - 'community' : largest community of nodes
      - 'one_side' : nodes on one geometric side based on positions
      - 'half_graph' : BFS from a random start until ~half the nodes are collected
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    deg_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    x0 = np.zeros(n)
    label = scenario

    if scenario == "single_hub":
        hub = deg_sorted[0][0]
        x0[nodes.index(hub)] = 1.0
        label = f"single_hub_{hub}"
    elif scenario == "top_k_hubs":
        hubs = [d[0] for d in deg_sorted[:k]]
        for h in hubs:
            x0[nodes.index(h)] = 1.0 / len(hubs)
        label = f"top_{k}_hubs"
    elif scenario == "cluster_hub":
        hub = deg_sorted[0][0]
        nbrs = list(G.neighbors(hub))
        seeds = [hub] + nbrs[:k]
        for s in seeds:
            x0[nodes.index(s)] = 1.0 / len(seeds)
        label = f"cluster_{hub}"
    elif scenario == "random_k":
        picks = list(rng.choice(nodes, size=min(k, n), replace=False))
        for p in picks:
            x0[nodes.index(p)] = 1.0 / len(picks)
        label = f"random_{len(picks)}"
    elif scenario == "k_hop_area":
        # pick the top hub and include nodes within `radius` hops
        hub = deg_sorted[0][0]
        lengths = nx.single_source_shortest_path_length(G, hub, cutoff=radius)
        picks = list(lengths.keys())
        for p in picks:
            x0[nodes.index(p)] = 1.0 / len(picks)
        label = f"k_hop_{hub}_r{radius}"
    elif scenario == "community":
        # detect communities and heat an entire community (largest by size)
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except Exception:
            # fallback: treat top hub neighborhood as community
            hub = deg_sorted[0][0]
            communities = [set([hub] + list(G.neighbors(hub)))]
        # pick largest community
        largest = max(communities, key=len)
        picks = list(largest)
        for p in picks:
            x0[nodes.index(p)] = 1.0 / len(picks)
        label = f"community_{len(picks)}nodes"
    elif scenario == "one_side":
        # use layout positions to pick one geometric side (e.g., x < median)
        if pos is None:
            raise ValueError(
                "'one_side' scenario requires `pos` argument (node positions)"
            )
        xs = np.array([pos[n][0] for n in nodes])
        med = np.median(xs)
        picks = [n for n, x in zip(nodes, xs) if x < med]
        for p in picks:
            x0[nodes.index(p)] = 1.0 / len(picks)
        label = f"one_side_{len(picks)}nodes"
    elif scenario == "half_graph":
        # BFS from a random start until we've collected ~half the nodes
        start = rng.choice(nodes)
        collected = []
        for u in nx.bfs_tree(G, start):
            collected.append(u)
            if len(collected) >= n // 2:
                break
        for p in collected:
            x0[nodes.index(p)] = 1.0 / len(collected)
        label = f"half_from_{start}_{len(collected)}nodes"
    elif scenario == "distributed":
        # assign small heat to a fraction of nodes
        count = max(1, int(np.round(fraction * n)))
        picks = list(rng.choice(nodes, size=count, replace=False))
        for p in picks:
            x0[nodes.index(p)] = rng.random() * 0.5 + 0.1
        # normalize
        if x0.sum() > 0:
            x0 = x0 / x0.sum()
        label = f"distributed_{int(fraction*100)}pct"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return x0, label


def run_scenarios(G, pos, scenarios, t_max=5.0, steps=60, output_dir="."):
    times_shared = None
    records = {}
    plot_indices = [0, max(1, int(0.1 * steps)), int(0.5 * steps), steps - 1]
    for scen in scenarios:
        scenario_name = scen.get("name") if isinstance(scen, dict) else scen
        params = scen.get("params", {}) if isinstance(scen, dict) else {}
        x0, label = make_initial_state(G, scenario=scenario_name, pos=pos, **params)
        times, heat_matrix, init_idx = simulate_heat_equation(
            G, initial_state=x0, t_max=t_max, steps=steps
        )
        times_shared = times
        # Save traces and snapshots per scenario
        safe_label = label.replace(" ", "_")
        trace_file = os.path.join(output_dir, f"heat_traces_{safe_label}.png")
        plot_heat_traces(
            times,
            heat_matrix,
            title=f"Heat Traces ({safe_label})",
            out_filename=trace_file,
        )
        snapshots_file = os.path.join(output_dir, f"heat_snapshots_{safe_label}.png")
        plot_graph_snapshots(
            G,
            pos,
            times,
            heat_matrix,
            plot_indices,
            title=f"Snapshots ({safe_label})",
            out_filename=snapshots_file,
        )
        records[safe_label] = {"times": times, "heat": heat_matrix, "label": safe_label}
    # Optionally: produce a combined comparison of source-average heat
    comp_file = os.path.join(output_dir, "heat_traces_scenarios_comparison.png")
    plt.figure(figsize=(12, 7))
    for lbl, rec in records.items():
        # plot average heat across nodes as a simple comparison
        avg_heat = rec["heat"].mean(axis=1)
        plt.plot(times_shared, avg_heat, label=lbl)
    plt.title(
        "Average Heat Over Time: Scenario Comparison", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Time (t)")
    plt.ylabel("Average Heat")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(comp_file, dpi=150)
    print(f"Saved scenario comparison traces to {comp_file}")
    plt.close()
    return records


def plot_multiple_snapshots(
    G,
    pos,
    time_steps,
    heat_matrices,
    initial_nodes,
    indices_to_plot,
    title,
    out_filename,
):
    n_sims = len(heat_matrices)
    n_times = len(indices_to_plot)
    fig, axes = plt.subplots(n_times, n_sims, figsize=(5 * n_sims, 4 * n_times))
    axes = np.atleast_2d(axes)
    # global max heat for consistent color mapping
    max_heat = max(np.max(hm) for hm in heat_matrices)
    nodes_artist = None
    for col, hm in enumerate(heat_matrices):
        for row, idx in enumerate(indices_to_plot):
            ax = axes[row, col]
            node_colors = hm[idx, :]
            nodes_artist = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=node_colors,
                cmap=plt.cm.inferno,
                node_size=80,
                vmin=0,
                vmax=max_heat,
                edgecolors="black",
                linewidths=0.5,
            )
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")
            if row == 0:
                ax.set_title(f"Source: {initial_nodes[col]}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"t = {time_steps[idx]:.1f}", fontsize=10)
            ax.axis("off")
    plt.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    cbar = fig.colorbar(
        nodes_artist,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.8,
        pad=0.05,
    )
    cbar.set_label("Heat Intensity", fontsize=12)
    plt.savefig(out_filename, dpi=150, bbox_inches="tight")
    print(f"Saved combined snapshots to {out_filename}")
    plt.close()


def run_simulations(G, initial_nodes, t_max=5.0, steps=60, pos=None, output_dir="."):
    times_shared = None
    heat_matrices = []
    init_indices = []
    plot_indices = [0, max(1, int(0.1 * steps)), int(0.5 * steps), steps - 1]
    for initial_node in initial_nodes:
        times, heat_matrix, init_idx = simulate_heat_equation(
            G, initial_node=initial_node, t_max=t_max, steps=steps
        )
        times_shared = times
        heat_matrices.append(heat_matrix)
        init_indices.append(init_idx)
        # Individual outputs
        trace_file = os.path.join(
            output_dir, f"heat_kernel_traces_source_{initial_node}.png"
        )
        plot_heat_traces(
            times,
            heat_matrix,
            title=f"Heat Diffusion Traces (source={initial_node})",
            out_filename=trace_file,
        )
        snapshots_file = os.path.join(
            output_dir, f"heat_kernel_snapshots_source_{initial_node}.png"
        )
        plot_graph_snapshots(
            G,
            pos,
            times,
            heat_matrix,
            plot_indices,
            title=f"Heat Spread (source={initial_node})",
            out_filename=snapshots_file,
        )
    # Combined comparison: source node traces
    combined_file = os.path.join(output_dir, "heat_kernel_traces_comparison.png")
    plt.figure(figsize=(12, 7))
    for init_node, hm, idx in zip(initial_nodes, heat_matrices, init_indices):
        plt.plot(times_shared, hm[:, idx], label=f"source {init_node}", linewidth=2)
    plt.title("Comparison: Source Node Heat Over Time", fontsize=16, fontweight="bold")
    plt.xlabel("Time (t)", fontsize=14)
    plt.ylabel("Heat at Source Node", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(combined_file, dpi=150)
    print(f"Saved combined traces to {combined_file}")
    plt.close()
    # Combined snapshot grid
    grid_file = os.path.join(output_dir, "heat_kernel_snapshots_comparison.png")
    plot_multiple_snapshots(
        G,
        pos,
        times_shared,
        heat_matrices,
        initial_nodes,
        plot_indices,
        title="Comparative Snapshots",
        out_filename=grid_file,
    )
    return times_shared, heat_matrices


def main():
    start_time = time.time()

    # 1. Generate a Graph
    print("Generating a Barabasi-Albert network (n=100, m=3)...")
    G = nx.barabasi_albert_graph(n=100, m=3, seed=42)

    # Use spring layout or kamada_kawai for nice visualization
    pos = nx.spring_layout(G, seed=42, k=0.15)

    # 2. Define scenarios to test multiple initial conditions
    scenarios = [
        {"name": "single_hub", "params": {}},
        {"name": "top_k_hubs", "params": {"k": 3}},
        {"name": "k_hop_area", "params": {"radius": 2}},
        {"name": "community", "params": {}},
        {"name": "one_side", "params": {}},
        {"name": "half_graph", "params": {}},
        {"name": "distributed", "params": {"fraction": 0.1, "seed": 42}},
    ]

    # Ensure Week 7 directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # 3. Run scenario experiments and save visualizations
    records = run_scenarios(
        G, pos, scenarios, t_max=5.0, steps=60, output_dir=output_dir
    )

    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
