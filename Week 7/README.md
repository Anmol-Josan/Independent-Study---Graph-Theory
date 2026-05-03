# Week 7: Diffusion & Heat Kernels

## Task 1: The Heat Equation on Graphs
The **Heat Equation** models how a quantity (like heat, a disease, or a rumor) diffuses over a structure over time. On a graph, the equation is driven by the **Graph Laplacian**, $L$, which encodes the network's structure:

$$ \frac{d}{dt} H_t = -L H_t $$

The solution to this differential equation gives us the **Heat Kernel** (the operator that pushes the heat distribution forward in time):
$$ H_t = e^{-tL} $$

Where:
- $L = D - A$ (Degree Matrix minus Adjacency Matrix)
- $H_t$ is a matrix describing how heat moves from any node to any other node over a time interval $t$.
- If we start with an initial concentration of heat (or an initial outbreak) represented by a vector $x_0$, the state at time $t$ is simply: $x_t = H_t x_0 = e^{-tL} x_0$.

## Task 2: Implementing the Simulation
I have created the graph simulation in `heat_kernel.py`. It constructs a scale-free network (Barabási–Albert model, representing many real-world systems), drops an initial "concentration" of heat/disease on the most connected hub node, and computes how it diffuses using the matrix exponential.

### Generated Visualizations (inside the `Week 7` folder):
1. **[heat_kernel_traces.png](heat_kernel_traces.png)**: 
   Plots the heat/spillover level per node across time steps. You will see the source node's concentration decay dramatically, while the remaining nodes experience a wave of heat that eventually plateaus as the entire network reaches thermal equilibrium.
2. **[heat_kernel_snapshots.png](heat_kernel_snapshots.png)**: 
   Visually maps the heat concentration as colors on the actual network topology across four different time slices, giving an intuition of *where* the spread goes first (usually to other densely connected hubs, before seeping into the peripheral leaf nodes).
   
## Multi-source simulations
The simulation script now runs several initial-source experiments in one go: two highest-degree hubs, a peripheral (leaf) node, and a random node. For each source the script saves:
- `heat_kernel_traces_source_<node>.png` — per-node heat traces over time for that source  
- `heat_kernel_snapshots_source_<node>.png` — spatial snapshots across time for that source  
It also creates comparison files: `heat_kernel_traces_comparison.png` and `heat_kernel_snapshots_comparison.png` to visually compare dynamics across different initial outbreak locations.
