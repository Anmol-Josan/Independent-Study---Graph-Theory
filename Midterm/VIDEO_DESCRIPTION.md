# Midterm Video Description

## Short Description (for submission)

This video demonstrates the Connectivity Analyzer workflow, which combines spectral connectivity (algebraic connectivity, lambda_2) with centrality metrics to identify structurally important nodes. For this demo, we use the Stanford SNAP Facebook network, extract a dense connected core, and animate targeted removals. The updated logic now selects a sequence that keeps lambda_2 non-increasing across removal steps, so the stability curve reflects cumulative structural damage without misleading rebounds.

## What the Video Shows

1. A connected Facebook subgraph is loaded and visualized with node coloring based on combined criticality.
2. The script prepares a second-tier candidate pool (after reserving the absolute top hubs) and picks up to five attack targets greedily.
3. At each step, a node is accepted only if removing it keeps lambda_2 at or below the current value.
4. The animation alternates between:
   - A highlight frame (node marked in cyan, metric unchanged), and
   - A removal frame (node deleted, metric updated).
5. A bottom chart tracks lambda_2 after each confirmed removal.

## What It Means

- The network is dense and socially clustered, but still vulnerable to targeted structural attacks.
- Flat segments during highlight frames are expected because no node has been removed yet.
- The updated target logic keeps the removal trajectory non-increasing, so the curve does not bounce upward because of target-choice artifacts.
- Practically, this identifies members whose departure most weakens information flow across the broader network.
