import os
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


N_NODES = 100            
CLIQUE_SIZE = 9          
NUM_GRAPHS = 6          
EDGE_PROBABILITY = 0.8
AREA_SIDE = 5.0          
MAX_RADIUS = 1.0         
CLUSTER_FRACTION = 0.45  

OUTPUT_CSV = "csma_clique9_dataset.csv"
OUTPUT_PNG = "csma_clique9_visualization.png"


def get_seed(idx, p, clique_size):
    s = f"csma_cpu_v1_{clique_size}_{idx}_{p}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def pairwise_distances_cpu(positions):
    """Pure CPU NumPy pairwise Euclidean distance matrix."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)


def sample_cluster_positions(k, center, max_radius, side, rng, frac=CLUSTER_FRACTION):
    r = frac * max_radius
    pts = np.empty((k, 2), dtype=np.float64)
    for i in range(k):
        while True:
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, r)
            x = center[0] + rad * np.cos(ang)
            y = center[1] + rad * np.sin(ang)
            if 0.0 <= x <= side and 0.0 <= y <= side:
                pts[i] = (x, y)
                break
    return pts


def build_positions_and_plant(n, side, radius, clique_size, rng):
    positions = rng.uniform(0, side, size=(n, 2))
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    
    chosen = all_idx[:clique_size]
    center = rng.uniform(radius * CLUSTER_FRACTION, side - radius * CLUSTER_FRACTION, size=2)
    positions[chosen] = sample_cluster_positions(clique_size, center, radius, side, rng)
    
    return positions, chosen


def build_adjacency(positions, radius, p, clique_nodes, rng):
    n = positions.shape[0]
    dist = pairwise_distances_cpu(positions)

    within_radius = (dist <= radius) & (dist > 0)
    rand_draw = rng.random((n, n))
    rand_mask = np.triu(rand_draw < p, k=1)
    rand_mask = rand_mask | rand_mask.T

    adj = np.zeros((n, n), dtype=np.uint8)
    adj[within_radius & rand_mask] = 1

    protected_edges = set()
    g = list(clique_nodes)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g[i], g[j]
            adj[a, b] = 1
            adj[b, a] = 1
            protected_edges.add(frozenset((int(a), int(b))))

    np.fill_diagonal(adj, 0)
    return adj, protected_edges


def repair_max_clique(adj, target_size, protected_edges, rng, max_iters=8000):
    G = nx.from_numpy_array(adj)
    it = 0
    while it < max_iters:
        it += 1
        cliques = list(nx.find_cliques(G))
        biggest = max(cliques, key=len)
        if len(biggest) <= target_size:
            break
        removable = [
            (biggest[i], biggest[j])
            for i in range(len(biggest))
            for j in range(i + 1, len(biggest))
            if frozenset((biggest[i], biggest[j])) not in protected_edges
        ]
        if not removable:
            break
        u, v = removable[rng.integers(len(removable))]
        G.remove_edge(u, v)

    n = adj.shape[0]
    adj_fixed = nx.to_numpy_array(G, nodelist=range(n)).astype(np.uint8)
    actual_max = max(len(c) for c in nx.find_cliques(G))
    return adj_fixed, actual_max, it


def connect_isolated_nodes(adj, positions, radius, rng, side):
    n = adj.shape[0]
    dist = pairwise_distances_cpu(positions)
    degrees = adj.sum(axis=1)
    isolated = np.where(degrees == 0)[0]

    for node in isolated:
        d = dist[node].copy()
        d[node] = np.inf
        nearest = int(np.argmin(d))

        if d[nearest] > radius:
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, radius * 0.9)
            new_x = float(np.clip(positions[nearest, 0] + r * np.cos(angle), 0.0, side))
            new_y = float(np.clip(positions[nearest, 1] + r * np.sin(angle), 0.0, side))
            positions[node] = (new_x, new_y)

        adj[node, nearest] = 1
        adj[nearest, node] = 1

    return adj, positions


def main():
    unique_graphs = []
    unique_positions = []
    unique_clique_nodes = []
    unique_wl_hashes = set()
    dataset_rows = []

    attempts = 0
    while len(unique_graphs) < NUM_GRAPHS and attempts < 2000:
        attempts += 1
        seed = get_seed(attempts, EDGE_PROBABILITY, CLIQUE_SIZE)
        rng = np.random.default_rng(seed)
        
        positions, clique_nodes = build_positions_and_plant(N_NODES, AREA_SIDE, MAX_RADIUS, CLIQUE_SIZE, rng)
        adj, protected_edges = build_adjacency(positions, MAX_RADIUS, EDGE_PROBABILITY, clique_nodes, rng)
        adj, actual_max, _ = repair_max_clique(adj, CLIQUE_SIZE, protected_edges, rng)
        adj, positions = connect_isolated_nodes(adj, positions, MAX_RADIUS, rng, AREA_SIDE)
        
        if actual_max != CLIQUE_SIZE:
            continue
            
        G = nx.from_numpy_array(adj)
        wl_hash = nx.weisfeiler_lehman_graph_hash(G, iterations=3)
        
        if wl_hash in unique_wl_hashes:
            continue
            
        unique_wl_hashes.add(wl_hash)
        unique_graphs.append(G)
        unique_positions.append(positions)
        unique_clique_nodes.append(clique_nodes)
        
        row = {
            "graph_id": len(unique_graphs) - 1,
            "clique_size": CLIQUE_SIZE,
            "p": EDGE_PROBABILITY,
            "num_cliques": 1,
            "num_nodes": N_NODES,
            "area_side": AREA_SIDE,
            "max_radius": MAX_RADIUS,
            "actual_max_clique": actual_max
        }
        for i in range(N_NODES):
            for j in range(N_NODES):
                row[f"edge_{i}_{j}"] = adj[i, j]
                
        dataset_rows.append(row)

    df_export = pd.DataFrame(dataset_rows)
    df_export.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset saved to '{OUTPUT_CSV}'.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx in range(len(unique_graphs)):
        G = unique_graphs[idx]
        pos = {i: unique_positions[idx][i] for i in range(N_NODES)}
        clique_nodes = unique_clique_nodes[idx]
        
        ax = axes[idx]
        node_colors = ['#FF3333' if node in clique_nodes else '#72BCD4' for node in G.nodes()]
        node_sizes = [28 if node in clique_nodes else 10 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color='gray')
        
        clique_subgraph = G.subgraph(clique_nodes)
        nx.draw_networkx_edges(clique_subgraph, pos, ax=ax, alpha=0.85, edge_color='red', width=1.5)
        
        ax.set_title(f"Unique Graph {idx} (p={EDGE_PROBABILITY}, Max Clique={CLIQUE_SIZE})", fontsize=10, fontweight='bold')
        ax.set_xlim(-0.1, AREA_SIDE + 0.1)
        ax.set_ylim(-0.1, AREA_SIDE + 0.1)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()
    print(f"Plot visualization saved to '{OUTPUT_PNG}'.")

if __name__ == "__main__":
    main()