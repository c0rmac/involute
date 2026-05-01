import numpy as np
import itertools
import pickle
import gc
import multiprocessing as mp
from numpy.polynomial.legendre import leggauss

# =============================================================================
# 1. HARDCODED PARAMETERS
# =============================================================================
n = 40
k = 20
M_LOCAL = 7         # 7 is now completely safe. 8 is possible if you have time!

#alpha_values = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0], dtype=np.float32)
alpha_values = np.array([0.01], dtype=np.float32)
k_eff = min(k, n - k)
SPLIT_DEPTH = 2     # We lock the first 2 angles to slice the RAM usage by M^2

M_SCOUT = max(15, k_eff + 4)

print(f"Initializing RAM-Safe Pipeline (M_local={M_LOCAL}, Cores={mp.cpu_count()})...")

base_nodes, base_weights = leggauss(M_LOCAL)

# =============================================================================
# 2. THE SCOUT (Unchanged)
# =============================================================================
def get_theta1_bounds(alpha):
    c_nodes, c_weights = leggauss(M_SCOUT)
    c_theta_1d = np.float32((c_nodes + 1) * np.pi / 4)
    c_weights_1d = np.float32(c_weights * np.pi / 4)
    c_indices = np.array(list(itertools.combinations(range(M_SCOUT), k_eff)))[:, ::-1]
    theta = c_theta_1d[c_indices]
    W_chunk = np.prod(c_weights_1d[c_indices], axis=1)

    short_power = np.float32(n - 2 * k_eff)
    log_weyl = np.zeros(len(c_indices), dtype=np.float32)
    if short_power > 0: log_weyl += short_power * np.sum(np.log(np.sin(theta)), axis=1)
    if k_eff > 1:
        for i in range(k_eff):
            for j in range(i + 1, k_eff):
                log_weyl += np.log(np.sin(theta[:, i] - theta[:, j]))
                log_weyl += np.log(np.sin(theta[:, i] + theta[:, j]))

    log_p = log_weyl - alpha * np.sum(theta**2, axis=1)
    global_peak = np.max(log_p)
    mass = np.exp(log_p - global_peak, dtype=np.float32) * W_chunk

    active_mask = mass > 1e-12
    if not np.any(active_mask): return global_peak, 0.0, np.pi / 2
    t1_min, t1_max = np.min(theta[active_mask, 0]), np.max(theta[active_mask, 0])
    span = t1_max - t1_min
    return global_peak, max(0.0, t1_min - 0.1 * span - 0.02), min(np.pi / 2, t1_max + 0.1 * span + 0.02)

# =============================================================================
# 3. WORKER: ISOLATED CHUNK PROCESSING
# =============================================================================
def build_raw_mass_tree(depth, start_idx, end_idx, paths, mass_cumsum, global_depth):
    """Recursively builds the dictionary using RAW MASS (not CDFs yet)."""
    node_mass = mass_cumsum[end_idx] - mass_cumsum[start_idx]
    if node_mass < 1e-18: return None
    if global_depth == k_eff: return {'_p': float(node_mass), 'children': {}}

    chunk_size = (end_idx - start_idx) // M_LOCAL
    children = {}
    total_p = 0.0

    for i in range(M_LOCAL):
        c_start = start_idx + i * chunk_size
        c_end = c_start + chunk_size
        if (mass_cumsum[c_end] - mass_cumsum[c_start]) >= 1e-18:
            t_val = float(paths[c_start, depth])
            child = build_raw_mass_tree(depth + 1, c_start, c_end, paths, mass_cumsum, global_depth + 1)
            if child:
                children[t_val] = child
                total_p += child['_p']

    if not children: return None
    return {'_p': float(total_p), 'children': children}

def process_subgrid(args):
    """Worker generates only a tiny fraction of the total paths."""
    prefix_angles, prefix_w, alpha, peak_log_p = args
    rem_k = k_eff - len(prefix_angles)
    bound = prefix_angles[-1]

    # 1. Generate the local micro-grid
    N = M_LOCAL ** rem_k
    paths = np.zeros((N, rem_k), dtype=np.float32)
    quad_w = np.ones(N, dtype=np.float32)

    span = bound / 2.0
    paths[:, 0] = np.repeat((base_nodes + 1) * span, M_LOCAL**(rem_k - 1))
    quad_w[:] = np.repeat(base_weights * span, M_LOCAL**(rem_k - 1))

    for d in range(1, rem_k):
        b = paths[:, d - 1] / 2.0
        repeats = M_LOCAL**(rem_k - 1 - d)
        tiles = M_LOCAL**d
        paths[:, d] = (np.tile(np.repeat(base_nodes, repeats), tiles) + 1) * b
        quad_w *= (np.tile(np.repeat(base_weights, repeats), tiles) * b)

    # 2. Glue the prefix onto the micro-grid for math evaluation
    full_paths = np.hstack([np.tile(prefix_angles, (N, 1)), paths])
    full_quad_w = quad_w * prefix_w

    # 3. Calculate Weyl and Mass
    short_power = np.float32(n - 2 * k_eff)
    log_weyl = np.zeros(N, dtype=np.float32)
    if short_power > 0: log_weyl += short_power * np.sum(np.log(np.sin(full_paths)), axis=1)
    if k_eff > 1:
        for i in range(k_eff):
            for j in range(i + 1, k_eff):
                log_weyl += np.log(np.sin(full_paths[:, i] - full_paths[:, j]))
                log_weyl += np.log(np.sin(full_paths[:, i] + full_paths[:, j]))

    log_p = log_weyl - alpha * np.sum(full_paths**2, axis=1)
    mass = np.exp(log_p - peak_log_p, dtype=np.float32) * full_quad_w

    # 4. Build local tree
    mass_cumsum = np.zeros(N + 1, dtype=np.float64)
    mass_cumsum[1:] = np.cumsum(mass)

    sub_tree = build_raw_mass_tree(0, 0, N, paths, mass_cumsum, global_depth=len(prefix_angles))
    return prefix_angles, sub_tree

# =============================================================================
# 4. MAIN THREAD: TASK GENERATOR & STITCHER
# =============================================================================
def finalize_and_normalize(node):
    """
    Converts raw mass dicts into sorted CDF arrays.
    Uses native Python sorting to avoid np.float32 KeyErrors.
    """
    # If the node has no children, it's a leaf; just return the mass
    if not node or not node.get('children'):
        return None

    # 1. Use native sorted() on the keys to keep them as standard Python floats
    vals_list = sorted(node['children'].keys(), reverse=True)

    # 2. Extract probabilities using the original keys
    probs_list = [node['children'][v]['_p'] for v in vals_list]

    # 3. Convert to arrays only AFTER retrieval
    vals = np.array(vals_list, dtype=np.float32)
    probs = np.array(probs_list, dtype=np.float32)

    # 4. Normalize
    cdf = np.cumsum(probs)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        # Safety for zero-mass branches
        cdf = np.zeros_like(probs)

    # 5. Recurse
    children = {v: finalize_and_normalize(node['children'][v]) for v in vals_list}

    return {
        'vals': vals,
        'cdf': cdf,
        'children': children
    }

if __name__ == '__main__':
    for alpha in alpha_values:
        print(f"\n--- Processing alpha = {alpha:.2f} ---")
        peak_log_p, low_b, high_b = get_theta1_bounds(alpha)

        # Build the Level 0 (Theta 1) and Level 1 (Theta 2) prefix chunks
        span1 = (high_b - low_b) / 2.0
        t1_nodes = (base_nodes + 1) * span1 + low_b
        w1_nodes = base_weights * span1

        tasks = []
        for i, t1 in enumerate(t1_nodes):
            span2 = t1 / 2.0
            t2_nodes = (base_nodes + 1) * span2
            w2_nodes = base_weights * span2
            for j, t2 in enumerate(t2_nodes):
                prefix = (float(t1), float(t2))
                prefix_w = float(w1_nodes[i] * w2_nodes[j])
                tasks.append((prefix, prefix_w, alpha, peak_log_p))

        print(f"  Dispatched {len(tasks)} isolated memory chunks...")

        # Multiprocessing Pool
        master_tree = {'_p': 0.0, 'children': {}}
        with mp.Pool(mp.cpu_count()) as pool:
            for prefix, sub_tree in pool.imap_unordered(process_subgrid, tasks):
                if sub_tree is None: continue

                # Force cast to standard Python float to ensure hash consistency
                t1, t2 = float(prefix[0]), float(prefix[1])

                if t1 not in master_tree['children']:
                    master_tree['children'][t1] = {'_p': 0.0, 'children': {}}

                master_tree['children'][t1]['children'][t2] = sub_tree
                master_tree['children'][t1]['_p'] += sub_tree['_p']
                master_tree['_p'] += sub_tree['_p']

        print("  Stitching and normalizing tree...")
        final_tree = finalize_and_normalize(master_tree)

        artifact_name = f"stiefel_seq_alpha_{alpha:.2f}_n{n}_k{k_eff}_M{M_LOCAL}.pkl"
        with open(artifact_name, 'wb') as f:
            pickle.dump({'n': n, 'k_eff': k_eff, 'M_local': M_LOCAL, 'alpha': float(alpha), 'sparse_tree': final_tree}, f)

        print(f"  Saved {artifact_name}.")
        gc.collect()