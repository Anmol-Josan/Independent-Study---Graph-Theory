import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
import os
import time

start_time = time.time()

def stress_test_graph(G):
    """
    Stress test a graph by iteratively removing the most central node
    and logging the drop in connectivity (using the size of the largest
    connected component as the metric).
    """
    G_test = G.copy()
    results = []
    
    # Function to calculate connectivity metric
    def get_connectivity(graph):
        if graph.number_of_nodes() == 0:
            return 0
        return max(len(c) for c in nx.connected_components(graph))

    initial_connectivity = get_connectivity(G_test)
    print(f"Initial Graph -> Nodes: {G_test.number_of_nodes()}, Largest Component: {initial_connectivity}")
    
    step = 0
    while G_test.number_of_nodes() > 0:
        # 1. Calculate centrality (approximate betweenness centrality for performance)
        k_approx = min(40, G_test.number_of_nodes())
        centrality = nx.betweenness_centrality(G_test, k=k_approx)
        
        # 2. Identify the most central node
        most_central_node = max(centrality, key=centrality.get)
        max_centrality_val = centrality[most_central_node]
        
        # 3. Remove the node
        G_test.remove_node(most_central_node)
        step += 1
        
        # 4. Measure new connectivity and the drop
        new_connectivity = get_connectivity(G_test)
        drop = initial_connectivity - new_connectivity
        
        # Log the iteration
        print(f"Step {step}: Removed Node {most_central_node} (Centrality: {max_centrality_val:.4f}) | "
              f"New Connectivity: {new_connectivity} | Total Drop: {drop}")
        
        results.append({
            'step': step,
            'removed_node': most_central_node,
            'centrality': max_centrality_val,
            'largest_component_size': new_connectivity
        })
        
        # Stop early if the graph is completely shattered into single nodes
        if new_connectivity <= 1:
            print("Graph fully fragmented.")
            break
            
    return results

def iterative_test(n, m, show_plots=True):
    # Allow a special case for a complete graph when m == 'complete'
    if m == 'complete':
        print(f"Generating Complete graph (n={n})...")
        G = nx.complete_graph(n)
        graph_label = f"Complete Graph (n={n})"
    else:
        # Generate a scale-free graph using Barabasi-Albert model (has clear hub nodes)
        print("Generating Barabasi-Albert graph...")
        G = nx.barabasi_albert_graph(n=n, m=m, seed=42)
        graph_label = f"Barabasi-Albert (n={n}, m={m})"
    
    print("\nStarting Stress Test...")
    # Save and show initial graph layout for visualization
    pos = nx.spring_layout(G, seed=42)
    degrees = dict(G.degree())
    node_sizes = [v * 15 + 30 for v in degrees.values()]
    node_colors = list(degrees.values())
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=pos, node_size=node_sizes, node_color=node_colors, 
            cmap=plt.cm.cividis, with_labels=True, font_size=7, 
            font_color="white", font_weight="bold", 
            linewidths=0.5, edge_color="gray", alpha=0.9)
    init_plot_path = f"initial_graph_{n}_{m}.png"
    plt.title(f"Initial Graph (n={n}, m={m})\nNode color & size represent degree", fontsize=14)
    plt.suptitle(f"Initial Graph Visualization — {graph_label}", fontsize=16, fontweight='bold')
    plt.xlabel("Layout X", fontsize=12)
    plt.ylabel("Layout Y", fontsize=12)
    plt.savefig(init_plot_path, dpi=150)
    print(f"Saved initial graph visualization to {init_plot_path}")
    if show_plots:
        try:
            plt.show()
        except Exception:
            pass

    results = stress_test_graph(G)
    
    # Plotting the connectivity drop
    steps = [r['step'] for r in results]
    connectivity = [r['largest_component_size'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, connectivity, marker='o', linestyle='-', color='r', markersize=4)
    plt.title(f"Network Stress Test (n={n}, m={m}): Targeted Attack", fontsize=14)
    plt.suptitle(f"Stress Test: Targeted Node Removal — {graph_label}", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Nodes Removed", fontsize=12)
    plt.ylabel("Size of Largest Connected Component", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Adding an annotation about early drop
    if len(steps) > 5:
        plt.annotate('Initial fragmentation', xy=(steps[5], connectivity[5]), 
                     xytext=(steps[5]+10, connectivity[5]+10),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plot_path = f"stress_test_survival_{n}_{m}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved stress test plot to {plot_path}")
    if show_plots:
        try:
            plt.show()
        except Exception:
            pass

    return results

if __name__ == "__main__":
    combos = []
    args = []
    for n in [100, 200, 500]:  # Number of nodes
        for m in [2, 4, 6]:  # Number of edges to attach from a new node to existing nodes
            combos.append((n, m))
            args.append((n, m))
    # Add a fully connected (complete) graph test for n=500
    combos.append((500, 'complete'))
    args.append((500, 'complete'))

    # Run tests in parallel to use more compute; avoid interactive plotting in workers
    cpu_count = os.cpu_count() or 1
    print(f"\nRunning {len(args)} jobs using up to {cpu_count} processes...")
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # map accepts iterable of tuples; use starmap to pass multiple args
        results_list = pool.starmap(iterative_test, [(n, m, False) for (n, m) in args])

    all_series = {}
    keys = []
    for (n, m), res in zip(args, results_list):
        key = f"n{n}_m{m}"
        steps = [r['step'] for r in res]
        connectivity = [r['largest_component_size'] for r in res]
        all_series[key] = (steps, connectivity)
        keys.append(key)

    # Create a final comparison figure showing survival curves for all runs
    plt.figure(figsize=(14, 9))
    
    # Optional: use a colormap if we have many lines
    colors = plt.cm.tab10(range(len(keys)))
    
    for idx, key in enumerate(keys):
        steps, connectivity = all_series[key]
        # Normalize the connectivity drop (percentage) for better side-by-side scale comparison
        max_c = connectivity[0] if connectivity else 1
        pct_connectivity = [c / max_c * 100 for c in connectivity]
        
        # We plot percentage to compare networks of different initial nodes seamlessly
        plt.plot(steps, pct_connectivity, marker='', linestyle='-', linewidth=2, 
                 color=colors[idx % 10], label=key.replace('_', ', '))
        
    plt.title("Comparison: Network Resilience under Targeted Attack", fontsize=16, fontweight='bold')
    plt.suptitle(f"Comparison of Survival Curves", fontsize=18, fontweight='bold')
    plt.xlabel("Number of Nodes Removed", fontsize=14)
    plt.ylabel("Largest Connected Component (% of Initial Size)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Network Config (n, m)", fontsize=10, title_fontsize=12)
    
    comparison_path = "stress_test_comparison.png"
    plt.savefig(comparison_path, dpi=200)
    print(f"\nSaved comparison plot to {comparison_path}")

print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")