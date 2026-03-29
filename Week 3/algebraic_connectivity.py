import networkx as nx
import numpy as np

def second_smallest_eigenvalue(G):
    """
    Calculates the second smallest eigenvalue of the Laplacian matrix of a graph G.
    This value is also known as the algebraic connectivity or Fiedler value.
    """
    # Get the Laplacian matrix as a dense numpy matrix
    L = nx.laplacian_matrix(G).todense()
    
    # Calculate all eigenvalues
    eigenvalues = np.linalg.eigvals(L)
    
    # Sort eigenvalues in ascending order
    # The smallest eigenvalue of a Laplacian is always 0
    eigenvalues.sort()
    
    if len(eigenvalues) >= 2:
        return np.real(eigenvalues[1])
    else:
        return 0.0

if __name__ == "__main__":
    print("Testing Laplacian Eigenvalues (Algebraic Connectivity):\n")

    # 1. Disconnected Graph
    # Creating a graph consisting of two separate triangles
    G_disconnected = nx.Graph()
    nx.add_cycle(G_disconnected, [0, 1, 2])
    nx.add_cycle(G_disconnected, [3, 4, 5])
    
    lambda_2_disc = second_smallest_eigenvalue(G_disconnected)
    print(f"Graph Type: Disconnected (Two separate cycles)")
    print(f"Second Smallest Eigenvalue (\u03bb\u2082): {lambda_2_disc:.4f}")
    print("Interpretation: A value close to 0 indicates the graph is disconnected or very easily cut.\n")

    # 2. Highly Connected Graph
    # Creating a complete graph of 6 nodes (every node connected to every other node)
    G_connected = nx.complete_graph(6)
    
    lambda_2_conn = second_smallest_eigenvalue(G_connected)
    print(f"Graph Type: Highly Connected (Complete Graph K_6)")
    print(f"Second Smallest Eigenvalue (\u03bb\u2082): {lambda_2_conn:.4f}")
    print("Interpretation: A high value indicates the graph is highly connected and difficult to cut.")