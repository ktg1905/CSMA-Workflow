import pandas as pd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time


dataset_path = "csma_clique9_dataset_v3.csv"  
df = pd.read_csv(dataset_path)

graph_df = df[df['graph_id'] == 1].iloc[0]

V = int(graph_df['num_nodes']) if 'num_nodes' in graph_df else 100
print(f"Loaded Graph 1 with V = {V} nodes.")

edge_cols = [f'edge_{i}_{j}' for i in range(100) for j in range(100)]
adj_flat = graph_df[edge_cols].to_numpy(dtype=np.float32)
adj_full = adj_flat.reshape((100, 100))

adj = adj_full[:V, :V].copy()
np.fill_diagonal(adj, 0.0)

max_deg = int(adj.sum(axis=1).max())
neighbor_idx = -np.ones((V, max_deg), dtype=np.int64)
neighbor_cnt = np.zeros(V, dtype=np.int64)
for i in range(V):
    nbrs = np.nonzero(adj[i])[0]
    neighbor_idx[i, :len(nbrs)] = nbrs
    neighbor_cnt[i] = len(nbrs)

node_degrees = adj.sum(axis=1)

mu_target = np.full(V, 0.8 * (1.0 / 9.0), dtype=np.float32)


@njit(cache=True, fastmath=True)
def simulate_csma_numba(neighbor_idx, neighbor_cnt, p_active, steps, seed):
    np.random.seed(seed)
    V = p_active.shape[0]
    x = np.zeros(V, dtype=np.float32)
    c = np.zeros(V, dtype=np.float32)

    for t in range(steps):
        i = np.random.randint(0, V)

        s = 0.0
        for k in range(neighbor_cnt[i]):
            j = neighbor_idx[i, k]
            s += x[j]

        if s == 0.0:
            if np.random.random() < p_active[i]:
                x[i] = 1.0
            else:
                x[i] = 0.0
        else:
            x[i] = 0.0

        c += x

    return c / steps

_ = simulate_csma_numba(neighbor_idx, neighbor_cnt,
                         np.full(V, 0.5, dtype=np.float32), 10, 0)


max_iterations = 3000     
csma_steps = 100000       

alpha0 = 0.25          # initial learning rate
decay_rate = 400.0     # controls how fast alpha shrinks over iterations
ema_beta = 0.85        # smoothing factor for mu (higher = smoother/slower)
epsilon = 0.01         # convergence threshold on smoothed max error

rng = np.random.default_rng(0)
r = rng.random(V).astype(np.float32)

max_errors = []
r_history = []
mu_history = []

mu_smooth = np.zeros(V, dtype=np.float32)
first_update = True
error = np.inf
iter_num = 0

start_time = time.time()
print("\nStarting Stochastic Approximation Optimization...")

while error > epsilon and iter_num < max_iterations:
    iter_num += 1

    p_active = (1.0 / (1.0 + np.exp(-r))).astype(np.float32)  # sigmoid

    mu = simulate_csma_numba(neighbor_idx, neighbor_cnt, p_active,
                              csma_steps, iter_num)

    if first_update:
        mu_smooth = mu.copy()
        first_update = False
    else:
        mu_smooth = ema_beta * mu_smooth + (1 - ema_beta) * mu

    error = float(np.max(np.abs(mu_smooth - mu_target)))
    max_errors.append(error)

    r_history.append(r.copy())
    mu_history.append(mu.copy())

    alpha = alpha0 / (1.0 + iter_num / decay_rate)
    r = r + alpha * (mu_target - mu_smooth)

    if iter_num % 50 == 0:
        print(f"Iteration {iter_num:4d}/{max_iterations} | "
              f"Max Error: {error:.6f} | alpha: {alpha:.5f} | "
              f"Elapsed Time: {time.time() - start_time:.2f}s")

sgd_iterations = iter_num  

if error <= epsilon:
    print(f"\nConverged: error {error:.6f} < epsilon {epsilon} "
          f"after {iter_num} iterations.")
else:
    print(f"\nStopped at max_iterations ({max_iterations}) "
          f"without reaching epsilon. Final error: {error:.6f}")

print(f"Optimization Finished in {time.time() - start_time:.2f} seconds.")


final_r = r
final_mu = mu_smooth  
target_mu_np = mu_target
final_errors = np.abs(final_mu - target_mu_np)

results_df = pd.DataFrame({
    'node_id': np.arange(V),
    'final_r': final_r,
    'simulated_mu': final_mu,
    'target_mu': target_mu_np,
    'absolute_error': final_errors,
    'degree': node_degrees.astype(int)
})

results_df.to_csv("csma_node_results.csv", index=False)
print("Results successfully saved to 'csma_node_results.csv'.")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(range(1, sgd_iterations + 1), max_errors, color='crimson', linewidth=1.5)
axs[0, 0].axhline(epsilon, color='gray', linestyle=':', linewidth=1, label=f'epsilon = {epsilon}')
axs[0, 0].set_title("Max Error Convergence vs. Iterations")
axs[0, 0].set_xlabel("Iterations")
axs[0, 0].set_ylabel(r"Max Error ($|\mu_{target} - \mu|$)")
axs[0, 0].legend()
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)

axs[0, 1].scatter(np.arange(V), final_mu, alpha=0.7, color='teal', label='Simulated $\\mu_i$')
axs[0, 1].axhline(mu_target[0], color='r', linestyle='--', label=f'Target = {mu_target[0]:.4f}')
axs[0, 1].set_title("Per-Node Simulated Service Rate vs. Target")
axs[0, 1].set_xlabel("Node ID")
axs[0, 1].set_ylabel(r"Service Rate $\mu$")
axs[0, 1].legend()
axs[0, 1].grid(True, ls="--", alpha=0.5)

r_history_np = np.array(r_history)
for node_idx in range(min(10, V)):
    axs[1, 0].plot(range(1, sgd_iterations + 1), r_history_np[:, node_idx], label=f'Node {node_idx}')
axs[1, 0].set_title(r"Transmission Intensity ($r_i$) Trajectory (Sample Nodes)")
axs[1, 0].set_xlabel("Iterations")
axs[1, 0].set_ylabel(r"Intensity $r$")
axs[1, 0].grid(True, ls="--", alpha=0.5)

mu_history_np = np.array(mu_history)
for node_idx in range(min(10, V)):
    axs[1, 1].plot(range(1, sgd_iterations + 1), mu_history_np[:, node_idx], label=f'Node {node_idx}')
axs[1, 1].axhline(mu_target[0], color='gray', linestyle=':', linewidth=1, label='target')
axs[1, 1].set_title(r"Service Rate ($\mu_i$) Trajectory (Sample Nodes)")
axs[1, 1].set_xlabel("Iterations")
axs[1, 1].set_ylabel(r"Service Rate $\mu$")
axs[1, 1].grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("csma_convergence_plots.png", dpi=300)
plt.show()



verify_steps_per_replicate = 2_000_000   
verify_replicates = 10                   
                                          

p_active_final = (1.0 / (1.0 + np.exp(-final_r))).astype(np.float32)

verify_start = time.time()
print(f"\nRunning verification backtest: {verify_replicates} replicates x "
      f"{verify_steps_per_replicate:,} steps each...")

verify_runs = np.zeros((verify_replicates, V), dtype=np.float32)
for rep in range(verify_replicates):
    verify_runs[rep] = simulate_csma_numba(
        neighbor_idx, neighbor_cnt, p_active_final,
        verify_steps_per_replicate,
        seed=1_000_000 + rep,  # seeds disjoint from anything used in training
    )

verified_mu = verify_runs.mean(axis=0)
verified_mu_std = verify_runs.std(axis=0)
verified_mu_stderr = verified_mu_std / np.sqrt(verify_replicates)

print(f"Verification finished in {time.time() - verify_start:.2f} seconds.")

error_vs_target = np.abs(verified_mu - target_mu_np)
error_vs_training_estimate = np.abs(verified_mu - final_mu)

print(f"Verified max error vs target:            {error_vs_target.max():.6f}")
print(f"Verified max error vs training estimate: {error_vs_training_estimate.max():.6f}")

verified_results_df = pd.DataFrame({
    'node_id': np.arange(V),
    'degree': node_degrees.astype(int),
    'final_r': final_r,
    'target_mu': target_mu_np,
    'training_mu_estimate': final_mu,          # EMA-smoothed estimate from the SGD loop
    'verified_mu': verified_mu,                # long-run, low-noise ground truth
    'verified_mu_stderr': verified_mu_stderr,  # standard error across replicates
    'error_vs_target': error_vs_target,
    'error_vs_training_estimate': error_vs_training_estimate,
})

verified_results_df.to_csv("csma_verified_results.csv", index=False)
print("Verified results successfully saved to 'csma_verified_results.csv'.")


fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))

node_ids = np.arange(V)
axs2[0].errorbar(node_ids, verified_mu, yerr=verified_mu_stderr, fmt='o',
                  color='teal', ecolor='gray', capsize=2, markersize=3,
                  label='Verified $\\mu$ (long run)')
axs2[0].scatter(node_ids, final_mu, color='crimson', marker='x', s=20,
                 label='Training estimate $\\mu$')
axs2[0].axhline(target_mu_np[0], color='gray', linestyle=':', label='Target')
axs2[0].set_title("Verified vs. Training-Time $\\mu$ per Node")
axs2[0].set_xlabel("Node ID")
axs2[0].set_ylabel(r"Service Rate $\mu$")
axs2[0].legend()
axs2[0].grid(True, ls="--", alpha=0.5)

axs2[1].plot(node_ids, error_vs_target, 'o-', color='crimson',
             label='|verified - target|')
axs2[1].plot(node_ids, error_vs_training_estimate, 's--', color='steelblue',
             label='|verified - training estimate|')
axs2[1].set_title("Verification Errors per Node")
axs2[1].set_xlabel("Node ID")
axs2[1].set_ylabel("Absolute Error")
axs2[1].legend()
axs2[1].grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("csma_verification_plots.png", dpi=300)
plt.show()