import numpy as np
import itertools
import pickle
import gc
import multiprocessing as mp
from numpy.polynomial.legendre import leggauss

# =============================================================================
# 1. HARDCODED PARAMETERS
# =============================================================================
n = 20
k = 10
M = 15              # The high-resolution grid (Safe to use 40+ with multicore)
M_COARSE = 15       # The low-resolution scout
#alpha_values = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0], dtype=np.float32)
alpha_values = np.array([0.1], dtype=np.float32)

k_eff = min(k, n - k)
CHUNK_SIZE = 500000 # Feeds half a million combinations per CPU core per cycle

# =============================================================================
# 2. THE DUAL-ENDED SCOUT (Finds min and max probability boundaries)
# =============================================================================
def get_dual_scout(alpha):
    """
    Runs a tiny grid to find the global peak AND the tightest physical window
    [low_bound, high_bound] that contains the entire k-dimensional active mass.
    """
    c_nodes, c_weights = leggauss(M_COARSE)

    # Scout always scans the full fundamental Weyl Chamber [0, pi/2]
    c_theta_1d = np.float32((c_nodes + 1) * np.pi / 4)
    c_weights_1d = np.float32(c_weights * np.pi / 4)

    c_indices = np.array(list(itertools.combinations(range(M_COARSE), k_eff)))[:, ::-1]
    theta = c_theta_1d[c_indices]
    W_chunk = np.prod(c_weights_1d[c_indices], axis=1)

    # Calculate Weyl Kernel
    short_power = np.float32(n - 2 * k_eff)
    log_weyl = np.zeros(len(c_indices), dtype=np.float32)

    if short_power > 0:
        log_weyl += short_power * np.sum(np.log(np.sin(theta)), axis=1)
    if k_eff > 1:
        for i in range(k_eff):
            for j in range(i + 1, k_eff):
                log_weyl += np.log(np.sin(theta[:, i] - theta[:, j]))
                log_weyl += np.log(np.sin(theta[:, i] + theta[:, j]))

    theta_sq = np.sum(theta**2, axis=1)
    log_p = log_weyl - alpha * theta_sq

    global_peak = np.max(log_p)
    mass = np.exp(log_p - global_peak, dtype=np.float32) * W_chunk

    active_mask = mass > 1e-12
    if not np.any(active_mask):
        return global_peak, 0.0, np.pi / 2 # Fallback

    f_min = np.min(theta[active_mask])
    f_max = np.max(theta[active_mask])

    # Add a 10% safety buffer to BOTH ends, strictly capped at [0, pi/2]
    span = f_max - f_min
    low_bound = max(0.0, f_min - 0.1 * span - 0.02)
    high_bound = min(np.pi / 2, f_max + 0.1 * span + 0.02)

    return global_peak, low_bound, high_bound

# =============================================================================
# 3. MULTIPROCESSING WORKER & TASK GENERATOR
# =============================================================================
def process_chunk(args):
    """Isolated worker function to calculate geometry at C-speed via numpy."""
    chunk_indices, theta_1d, weights_1d, alpha, peak_log_p, n_val, k_eff_val = args

    indices = np.array(chunk_indices, dtype=np.int32)[:, ::-1]
    theta = theta_1d[indices]
    W_chunk = np.prod(weights_1d[indices], axis=1)

    short_power = np.float32(n_val - 2 * k_eff_val)
    log_weyl = np.zeros(len(indices), dtype=np.float32)

    if short_power > 0:
        log_weyl += short_power * np.sum(np.log(np.sin(theta)), axis=1)
    if k_eff_val > 1:
        for i in range(k_eff_val):
            for j in range(i + 1, k_eff_val):
                log_weyl += np.log(np.sin(theta[:, i] - theta[:, j]))
                log_weyl += np.log(np.sin(theta[:, i] + theta[:, j]))

    theta_sq = np.sum(theta**2, axis=1)
    log_p = log_weyl - alpha * theta_sq

    # Calculate density natively shifted by the Scout's peak to prevent underflow
    mass_chunk = np.exp(log_p - peak_log_p, dtype=np.float32) * W_chunk

    return indices, mass_chunk

def generate_tasks(M_val, k_val, chunk_size, theta_1d, weights_1d, alpha, peak_log_p, n_val):
    """Yields argument tuples to keep the multiprocessing pool constantly fed."""
    iterator = itertools.combinations(range(M_val), k_val)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk: break
        yield (chunk, theta_1d, weights_1d, alpha, peak_log_p, n_val, k_val)

def finalize_tree_to_cdf(node):
    """Recursively converts mass sums into normalized CDFs."""
    if not node: return {'vals': np.array([], dtype=np.int32), 'cdf': np.array([]), 'children': {}}
    vals = np.sort(np.array(list(node.keys()), dtype=np.int32))[::-1]
    probs = np.array([node[v]['_p'] for v in vals], dtype=np.float32)
    cdf = np.cumsum(probs)
    if cdf[-1] > 0: cdf /= cdf[-1]
    children = {v: finalize_tree_to_cdf(node[v]['children']) for v in vals if node[v]['children']}
    return {'vals': vals, 'cdf': cdf, 'children': children}

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print(f"Initializing Adaptive Multicore Pipeline (n={n}, k_eff={k_eff}, M={M})...")
    base_nodes, base_weights = leggauss(M)

    for alpha in alpha_values:
        print(f"\n--- Processing alpha = {alpha:.2f} ---")

        # 1. Scout the manifold
        peak_log_p, low_bound, high_bound = get_dual_scout(alpha)
        print(f"  Scout Report: Global Peak = {peak_log_p:.2f}")
        print(f"  Scout Report: Active Window = [{low_bound:.4f}, {high_bound:.4f}] radians")

        # 2. Scale the grid exactly to the active window
        span_half = (high_bound - low_bound) / 2
        mapped_theta_1d = np.float32((base_nodes + 1) * span_half + low_bound)
        mapped_weights_1d = np.float32(base_weights * span_half)

        tree = {}
        task_generator = generate_tasks(M, k_eff, CHUNK_SIZE, mapped_theta_1d, mapped_weights_1d, alpha, peak_log_p, n)

        print(f"  Firing up {mp.cpu_count()} CPU cores...")
        chunk_counter = 0

        # 3. Multiprocessing Loop
        with mp.Pool(mp.cpu_count()) as pool:
            for indices, mass_chunk in pool.imap_unordered(process_chunk, task_generator):

                # Main thread builds the sparse tree as data returns from workers
                for i in range(len(indices)):
                    mass_val = float(mass_chunk[i])
                    if mass_val < 1e-15: continue

                    curr = tree
                    for val in indices[i]:
                        if val not in curr: curr[val] = {'_p': 0.0, 'children': {}}
                        curr[val]['_p'] += mass_val
                        curr = curr[val]['children']

                chunk_counter += 1
                if chunk_counter % 20 == 0:
                    print(f"  Processed {chunk_counter * CHUNK_SIZE:,} combinations...")

        # 4. Finalize and Save
        print("  Finalizing CDFs...")
        final_tree = finalize_tree_to_cdf(tree)

        artifact_name = f"stiefel_gauss_alpha_{alpha:.2f}_n{n}_k{k_eff}_M{M}.pkl"
        payload = {
            'n': n, 'k_eff': k_eff, 'M': M, 'alpha': float(alpha),
            'theta_grid': mapped_theta_1d,
            'sparse_tree': final_tree
        }

        with open(artifact_name, 'wb') as f:
            pickle.dump(payload, f)

        print(f"  Saved {artifact_name}.")

        # Free memory immediately before starting the next alpha
        del tree, final_tree, payload
        gc.collect()

    print("\nGeneration Complete.")