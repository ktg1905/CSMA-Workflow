import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


NUM_NODES = 100          
CLIQUE_SIZE = 9          
NUM_CLIQUES = 1          
BASE_EDGE_PROB = 0.15    
SQUARE_SIDE = 5.0        
MAX_RADIUS = 1.0         
NUM_GRAPHS = 6           

def generate_spatial_constrained_clique_graph(num_nodes=NUM_NODES, clique_size=CLIQUE_SIZE, 
                                               p=BASE_EDGE_PROB, side_length=SQUARE_SIDE, max_radius=MAX_RADIUS,
                                               max_attempts=2000):
    
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        pos = {i: np.random.uniform(0, side_length, size=2) for i in range(num_nodes)}
        
        dist_G = nx.Graph()
        dist_G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(pos[i] - pos[j]) <= max_radius:
                    dist_G.add_edge(i, j)
                    

        valid_cliques = [c for c in nx.find_cliques(dist_G) if len(c) >= clique_size]
        
        if len(valid_cliques) > 0:
            chosen_cluster = np.random.choice(len(valid_cliques))
            clique_nodes = np.random.choice(valid_cliques[chosen_cluster], size=clique_size, replace=False)
            
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if dist_G.has_edge(i, j) and np.random.rand() < p:
                        G.add_edge(i, j)
                            
            for i in range(len(clique_nodes)):
                for j in range(i + 1, len(clique_nodes)):
                    G.add_edge(clique_nodes[i], clique_nodes[j])
                    
            return G, pos, clique_nodes
            
    raise RuntimeError("Could not find a valid spatial node cluster.")


def generate_unique_spatial_clique_graphs(num_target_graphs=NUM_GRAPHS, num_nodes=NUM_NODES, 
                                          clique_size=CLIQUE_SIZE, p=BASE_EDGE_PROB, 
                                          side_length=SQUARE_SIDE, max_radius=MAX_RADIUS, 
                                          max_total_attempts=500):
    
    unique_graphs = []
    unique_positions = []
    unique_clique_nodes = []
    wl_buckets = defaultdict(list)
    
    attempts = 0
    redundant_skipped = 0
    
    while len(unique_graphs) < num_target_graphs and attempts < max_total_attempts:
        attempts += 1
        G, pos, c_nodes = generate_spatial_constrained_clique_graph(
            num_nodes=num_nodes, clique_size=clique_size, p=p, 
            side_length=side_length, max_radius=max_radius
        )
        
        g_hash = nx.weisfeiler_lehman_graph_hash(G, iterations=3)
        
        is_redundant = False
        for existing_G in wl_buckets[g_hash]:
            if nx.is_isomorphic(G, existing_G):
                is_redundant = True
                redundant_skipped += 1
                break
                
        if not is_redundant:
            wl_buckets[g_hash].append(G)
            unique_graphs.append(G)
            unique_positions.append(pos)
            unique_clique_nodes.append(c_nodes)
            
    print(f"Created {len(unique_graphs)} unique graphs (skipped {redundant_skipped} redundant ones).")
    return unique_graphs, unique_positions, unique_clique_nodes


def main():

    graphs, positions, clique_nodes_list = generate_unique_spatial_clique_graphs()

    graphs_data = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for g_idx in range(len(graphs)):
        G = graphs[g_idx]
        pos = positions[g_idx]
        c_nodes = clique_nodes_list[g_idx]
        
        adj_matrix = nx.to_numpy_array(G, dtype=int)
        
        row = {
            "graph_id": g_idx,
            "clique_size": CLIQUE_SIZE,
            "p": BASE_EDGE_PROB,
            "num_cliques": NUM_CLIQUES,
            "square_side": SQUARE_SIDE,
            "max_radius": MAX_RADIUS,
            "num_nodes": NUM_NODES
        }
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                row[f"edge_{i}_{j}"] = adj_matrix[i, j]
                
        graphs_data.append(row)
        
        ax = axes[g_idx]
        node_colors = ['#FF3333' if node in c_nodes else '#72BCD4' for node in G.nodes()]
        node_sizes = [70 if node in c_nodes else 25 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color='gray')
        
        clique_subgraph = G.subgraph(c_nodes)
        nx.draw_networkx_edges(clique_subgraph, pos, ax=ax, alpha=0.9, edge_color='red', width=1.5)
        
        ax.set_title(f"Graph {g_idx + 1}, clique 9", fontsize=11, fontweight='bold')
        ax.set_xlim(0, SQUARE_SIDE)
        ax.set_ylim(0, SQUARE_SIDE)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("spatial_graphs.png", dpi=300)
    plt.show()

    df_dataset = pd.DataFrame(graphs_data)
    df_dataset.to_csv("spatial_clique_graph_dataset.csv", index=False)

if __name__=="__main__":
    main()