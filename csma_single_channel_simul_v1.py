import numpy as np
import pandas as pd
import networkx as nx


def simulate_csma(G, r, steps=10000):
    nodes = list(G.nodes())
    x = {i: 0 for i in nodes}
    c = {i: 0 for i in nodes}
    neighbors = {i: list(G.neighbors(i)) for i in nodes}
    
    for _ in range(steps):

        i = np.random.choice(nodes)
        
        if sum(x[j] for j in neighbors[i]) == 0:
            p_active = np.exp(r[i]) / (1.0 + np.exp(r[i]))
            x[i] = 1 if np.random.rand() < p_active else 0
        else:
            x[i] = 0
            
        for v in nodes:
            c[v] += x[v]
            
    mu = {i: c[i] / steps for i in nodes}
    
    return mu



def stochastic_approximation_update(G, target_mu, alpha_0=0.5, decay=0.01, epsilon=0.01, max_steps=10000, max_iters=300):
    
    nodes = list(G.nodes())
    
    if isinstance(target_mu, (int, float)):
        target_dict = {i: float(target_mu) for i in nodes}
    else:
        target_dict = target_mu

    r = {i: np.random.uniform(0.0, 1.0) for i in nodes}
    
    mu_ema = {}
    
    for it in range(max_iters):
        # Run CSMA simulation[cite: 1]
        mu = simulate_csma(G, r, steps=max_steps)
        
        
        for i in nodes:
            if i not in mu_ema:
                mu_ema[i] = mu[i]
            else:
                mu_ema[i] = 0.8 * mu_ema[i] + 0.2 * mu[i]
            
    
        alpha_t = alpha_0 / (1.0 + decay * it)
        
        for i in nodes:
            r[i] += alpha_t * (target_dict[i] - mu_ema[i])
            
            r[i] = min(max(r[i], -15.0), 15.0)
            
        error = max(abs(mu_ema[i] - target_dict[i]) for i in nodes)
        
        if error < epsilon:
            print(f"--> Converged in {it + 1} iterations (error = {error:.4f} < {epsilon})")
            break
    else:
        print(f"--> Reached max iterations ({max_iters}) with final error = {error:.4f}")

    return r, mu_ema



def generate_csma_pairs_dataset(input_csv="sample1.csv", output_csv="csma_pairs_dataset.csv", target_mu=0.03):
    print(f"Loading graphs from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    edge_cols = [c for c in df.columns if c.startswith("edge_")]
    edge_matrices = df[edge_cols].to_numpy(dtype=int)
    
    results = []
    
    for graph_idx in range(len(df)):
        print(f"Processing Graph {graph_idx + 1}/{len(df)}...")
        
        matrix_flat = edge_matrices[graph_idx]
        matrix_2d = matrix_flat.reshape((100, 100))
        G = nx.from_numpy_array(matrix_2d)
        
        graph_id = df.loc[graph_idx, "graph_id"] if "graph_id" in df.columns else graph_idx
        clique_size = df.loc[graph_idx, "clique_size"]
        p_param = df.loc[graph_idx, "p"]
        num_cliques = df.loc[graph_idx, "num_cliques"]
        
        r_opt, mu_achieved = stochastic_approximation_update(
            G, 
            target_mu=target_mu, 
            alpha_0=0.1, 
            decay = 0.01,
            epsilon=0.01, 
            max_steps=10000, 
            max_iters=1000
        )
        
        avg_r = round(float(np.mean(list(r_opt.values()))),4)
        avg_mu = round(float(np.mean(list(mu_achieved.values()))), 4)
        
        # Save one summary row per graph
        results.append({
            "graph_id": graph_id,
            "clique_size": clique_size,
            "p": p_param,
            "num_cliques": num_cliques,
            "avg_intensity_r": avg_r,
            "avg_service_rate_mu": avg_mu,
            "target_mu": target_mu
        })

        #Save summary per node of graph (r_i, mu_i)
        """
        for node_i in G.nodes():
            results.append({
                "graph_id": graph_idx,
                "node_id": node_i,
                "clique_size": clique_size,
                "p": p_param,
                "num_cliques": num_cliques,
                "node_degree": G.degree(node_i),
                "intensity_r": r_opt[node_i],
                "service_rate_mu": mu_achieved[node_i],
                "target_mu": target_mu
            })
        """
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print("Processing completed!")


def main():
    generate_csma_pairs_dataset()

if __name__=="__main__":
    main()