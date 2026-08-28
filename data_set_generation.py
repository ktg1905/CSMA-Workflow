import numpy as np
import networkx as nx
import pandas as pd
import glob
import os
from collections import defaultdict

# Generating graphs according to given clique size, number of cliques, number of nodes, edge probability and saving
# them to csv files

def generate_unique_clique_graphs_1(k, total_nodes=100, clique_size=10, p=0.1, max_attempts=1000):
    graphs = []
    matrices = []
    attempts = 0
    
    while len(matrices) < k and attempts < max_attempts:
        attempts += 1
        
        G = nx.erdos_renyi_graph(n=total_nodes, p=p)
        clique_nodes = np.random.choice(total_nodes, size=clique_size, replace=False)
        
        for i in range(len(clique_nodes)):
            for j in range(i + 1, len(clique_nodes)):
                G.add_edge(clique_nodes[i], clique_nodes[j])
                
        if any(nx.is_isomorphic(G, existing_g) for existing_g in graphs):
            continue  
            
        graphs.append(G)
        matrices.append(nx.to_numpy_array(G).astype(int))
        
    if len(matrices) < k:
        print(f"Warning: Only generated {len(matrices)} unique graphs after {max_attempts} attempts.")
        
    return matrices



def generate_unique_clique_graphs_2(k, total_nodes=100, clique_size=10, p=0.1, max_attempts=1000):
    
    graphs = []
    matrices = []
    attempts = 0
    
    if 2 * clique_size > total_nodes:
        raise ValueError(f"Two disjoint cliques of size {clique_size} require at least {2 * clique_size} nodes, but total_nodes={total_nodes}.")
    
    while len(matrices) < k and attempts < max_attempts:
        attempts += 1
        
        G = nx.erdos_renyi_graph(n=total_nodes, p=p)
        selected_nodes = np.random.choice(total_nodes, size=2 * clique_size, replace=False)
        
        clique1_nodes = selected_nodes[:clique_size]
        clique2_nodes = selected_nodes[clique_size:]
        
        for i in range(len(clique1_nodes)):
            for j in range(i + 1, len(clique1_nodes)):
                G.add_edge(clique1_nodes[i], clique1_nodes[j])
                
        for i in range(len(clique2_nodes)):
            for j in range(i + 1, len(clique2_nodes)):
                G.add_edge(clique2_nodes[i], clique2_nodes[j])
                
        if any(nx.is_isomorphic(G, existing_g) for existing_g in graphs):
            continue  
            
        graphs.append(G)
        matrices.append(nx.to_numpy_array(G).astype(int))
        
    if len(matrices) < k:
        print(f"Warning: Only generated {len(matrices)} unique graphs after {max_attempts} attempts.")
        
    return matrices


def convert_to_csv(matrices):
    flat_rows = [matrix.flatten() for matrix in matrices]

    col_names = [f"edge_{i}_{j}" for i in range(100) for j in range(100)]

    df = pd.DataFrame(flat_rows, columns=col_names)
    df.insert(0, "graph_id", range(len(matrices)))  # Add graph index column

    df.to_csv("<clique>_<edge_prob>_<no._of_clique>.csv", index=False)

#=============================================================================================================

#Combining files of same clique size

def combining_csv_same_clique_size():
    file_paths = glob.glob("<clique_size>_*.csv")
    all_dfs = []

    for path in sorted(file_paths):
        filename = os.path.basename(path).replace(".csv", "")
        parts = filename.split("_")
        
        clique_size = int(parts[0])
        p = float(parts[1])
        num_cliques = int(parts[2])
        
        df = pd.read_csv(path)
        
        df.insert(1, "clique_size", clique_size)
        df.insert(2, "p", p)
        df.insert(3, "num_cliques", num_cliques)
        
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df["graph_id"] = range(len(combined_df))

    combined_df.to_csv("clique_<clique_size>.csv", index=False)
    print(f"Successfully combined {len(file_paths)} files into shape: {combined_df.shape}")


#=================================================================================================================

# Combining files of different clique sizes into a single master dataset file, checking for redundancies and 
# sorting the graphs randomly

def process_and_combine_clique_csvs(file_pattern="clique_*.csv", output_file="master_clique_graphs.csv"):
    file_paths = glob.glob(file_pattern)
    all_rows = []

    print(f"Found {len(file_paths)} files to process...")

    for path in sorted(file_paths):
        filename = os.path.basename(path).replace(".csv", "")
        clique_size = int(filename.split("_")[1])
        
        df = pd.read_csv(path)
        
        edge_cols = [col for col in df.columns if col.startswith("edge_")]
        
        for _, row in df.iterrows():
            matrix_flat = np.asarray(row[edge_cols].values, dtype=int)
            matrix_2d = matrix_flat.reshape((100, 100))
            
            p = row["p"] if "p" in row else round(float(np.mean(matrix_2d)), 4)
            num_cliques = int(row["num_cliques"]) if "num_cliques" in row else 1
            
            all_rows.append({
                "clique_size": clique_size,
                "p": p,
                "num_cliques": num_cliques,
                "matrix_2d": matrix_2d,
                "matrix_flat": matrix_flat
            })

    print(f"Total graphs loaded across all files: {len(all_rows)}")

    buckets = defaultdict(list)
    unique_entries = []

    for entry in all_rows:
        G = nx.from_numpy_array(entry["matrix_2d"])
        g_hash = nx.weisfeiler_lehman_graph_hash(G, iterations=3)
        
        is_redundant = False
        for existing in buckets[g_hash]:
            existing_G = nx.from_numpy_array(existing["matrix_2d"])
            if nx.is_isomorphic(G, existing_G):
                is_redundant = True
                break
                
        if not is_redundant:
            buckets[g_hash].append(entry)
            unique_entries.append(entry)

    print(f"Graphs after removing redundant structures: {len(unique_entries)}")

    np.random.shuffle(unique_entries)

    edge_col_names = [f"edge_{i}_{j}" for i in range(100) for j in range(100)]
    
    final_rows = []
    for idx, entry in enumerate(unique_entries):
        row_dict = {
            "graph_id": idx,
            "clique_size": entry["clique_size"],
            "p": entry["p"],
            "num_cliques": entry["num_cliques"]
        }
        for col_name, val in zip(edge_col_names, entry["matrix_flat"]):
            row_dict[col_name] = val
            
        final_rows.append(row_dict)

    master_df = pd.DataFrame(final_rows)
    master_df.to_csv(output_file, index=False)
    print(f"Saved dataset to '{output_file}' with shape: {master_df.shape}")