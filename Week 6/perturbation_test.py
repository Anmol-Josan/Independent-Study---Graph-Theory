import networkx as nx
import matplotlib.pyplot as plt

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
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        return len(components[0])

    initial_connectivity = get_connectivity(G_test)
    print(f"Initial Graph -> Nodes: {G_test.number_of_nodes()}, Largest Component: {initial_connectivity}")
    
    step = 0
    while G_test.number_of_nodes() > 0:
        # 1. Calculate centrality (betweenness centrality is great for finding bottleneck nodes)
        centrality = nx.betweenness_centrality(G_test)
        
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

if __name__ == "__main__":
    # Generate a scale-free graph using Barabasi-Albert model (has clear hub nodes)
    print("Generating Barabasi-Albert graph...")
    G = nx.barabasi_albert_graph(n=100, m=3, seed=42)
    
    print("\nStarting Stress Test...")
    results = stress_test_graph(G)
    
    # Plotting the connectivity drop
    steps = [r['step'] for r in results]
    connectivity = [r['largest_component_size'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, connectivity, marker='o', linestyle='-', color='r')
    plt.title("Network Stress Test: Targeted Attack on Central Nodes")
    plt.xlabel("Number of Nodes Removed")
    plt.ylabel("Size of Largest Connected Component")
    plt.grid(True)
    
    plot_path = "stress_test_survival.png"
    plt.savefig(plot_path)
    print(f"\nSaved stress test plot to {plot_path}")