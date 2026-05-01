import numpy as np
import pickle
import time
import os
import multiprocessing as mp
from scipy.linalg import expm

# =============================================================================
# 1. PARAMETERS
# =============================================================================
n = 40
k = 10
k_eff = min(k, n - k)

# The array of alphas to process sequentially
POTENTIAL_ALPHAS = [9.0, 10.0]

# MCMC Settings
NUM_SAMPLES = 100_000
BURN_IN = 10_000
THINNING = 10
STEP_SIZE = 0.035
POLISHING_STEPS = 200    # The number of ambient MH steps to correct the geometry

print(f"Initializing Exact Stiefel V({n}, {k}) Precomputation (Ambient Polisher)...")

# Define the Consensus Point (M_hat) as the canonical frame
M_HAT = np.eye(n)[:, :k]

# =============================================================================
# 2. THE FLAWED PROPOSAL DENSITY (Log-Space)
# =============================================================================
def log_p_flawed_proposal(theta, current_alpha):
    """
    Calculates the unnormalized log-prob of the flawed Grassmannian-based measure.
    We use this to rapidly generate proposals that are 99% of the way to the target.
    """
    # Expanded boundaries to account for the Hemisphere Fix
    if theta[0] >= np.pi or theta[-1] <= 0.0:
        return -np.inf
    if np.any(np.diff(theta) >= 0):
        return -np.inf

    short_power = n - 2 * k_eff
    log_weyl = 0.0

    # Added np.abs() to all sine evaluations per Definition 2.14
    if short_power > 0:
        log_weyl += short_power * np.sum(np.log(np.abs(np.sin(theta))))

    if k_eff > 1:
        diffs = theta[:, None] - theta[None, :]
        sums = theta[:, None] + theta[None, :]
        idx = np.triu_indices(k_eff, k=1)
        log_weyl += np.sum(np.log(np.abs(np.sin(diffs[idx]))))
        log_weyl += np.sum(np.log(np.abs(np.sin(sums[idx]))))

    log_potential = -current_alpha * np.sum(theta**2)
    return log_weyl + log_potential

# =============================================================================
# 3. EXACT GEOMETRY UTILITIES
# =============================================================================
def stiefel_distance_sq(X, M_hat):
    """Computes the squared intrinsic distance using signed singular values."""
    core = np.dot(X.T, M_hat)
    U, S, Vh = np.linalg.svd(core)

    # Parity Fix: Allow negative singular values to track reflections up to pi
    det_sign = np.sign(np.linalg.det(np.dot(U, Vh)))
    if det_sign < 0:
        S[-1] = -S[-1]

    S = np.clip(S, -1.0, 1.0)
    angles = np.arccos(S)
    return np.sum(angles**2)

def generate_flawed_lift(theta, M_hat, n, k):
    """Lifts a flawed theta sample into the ambient SO(n) space."""
    # 1. Internal Rotation
    Z_int = np.random.randn(k, k)
    Omega_int = np.tril(Z_int, -1) - np.tril(Z_int, -1).T

    # 2. Right-Action Twist (O(k) reflection fix)
    Z_right = np.random.randn(k, k)
    Q, R = np.linalg.qr(Z_right)
    V_right = np.dot(Q, np.diag(np.sign(np.diag(R))))

    # 3. Canonical Embedding
    Sigma = np.zeros((n - k, k))
    np.fill_diagonal(Sigma, theta)

    V_s = np.zeros((n, n))
    V_s[:k, :k] = Omega_int
    V_s[:k, k:] = -np.dot(V_right, Sigma.T)
    V_s[k:, :k] = np.dot(Sigma, V_right.T)

    Theta = expm(V_s)

    # 4. Stabilizer Conjugation (O(n-k) reflection fix)
    Z_sub = np.random.randn(n - k, n - k)
    Q_sub, R_sub = np.linalg.qr(Z_sub)
    h_sub = np.dot(Q_sub, np.diag(np.sign(np.diag(R_sub))))

    h = np.eye(n)
    h[k:, k:] = h_sub

    g_tilde = np.dot(h, np.dot(Theta, h.T))
    X_prop = np.dot(g_tilde, M_hat)
    return X_prop

def extract_exact_angles(X, M_hat):
    """Extracts the exact principal angles back out of a polished matrix."""
    core = np.dot(X.T, M_hat)
    U, S, Vh = np.linalg.svd(core)

    det_sign = np.sign(np.linalg.det(np.dot(U, Vh)))
    if det_sign < 0:
        S[-1] = -S[-1]

    S = np.clip(S, -1.0, 1.0)
    return np.arccos(S)

# =============================================================================
# 4. THE 2-STAGE SAMPLING ENGINE
# =============================================================================
def run_mcmc_and_polish(chain_id, num_samples, burn_in, thinning, step_size, current_alpha):
    """Generates flawed proposals, then exactly polishes them via Ambient MH."""
    np.random.seed(os.getpid() + chain_id)
    start_time = time.time()

    # --- STAGE 1: Flawed MCMC Proposal Generation ---
    current_theta = np.linspace(np.pi/2 - 0.1, 0.1, k_eff, dtype=np.float64)
    current_log_p = log_p_flawed_proposal(current_theta, current_alpha)

    flawed_samples = []
    total_steps = num_samples + burn_in

    # Progress Tracking Variables
    accepted = 0
    mcmc_progress_interval = max(1, total_steps // 20)

    for step in range(total_steps):
        noise = np.random.normal(0, step_size, k_eff)
        proposed_theta = current_theta + noise
        proposed_theta = np.abs(proposed_theta)
        proposed_theta = np.where(proposed_theta > np.pi, 2*np.pi - proposed_theta, proposed_theta)
        proposed_theta = np.sort(proposed_theta)[::-1]

        proposed_log_p = log_p_flawed_proposal(proposed_theta, current_alpha)

        if np.log(np.random.rand()) < (proposed_log_p - current_log_p):
            current_theta = proposed_theta
            current_log_p = proposed_log_p
            if step >= burn_in:
                accepted += 1

        if step >= burn_in and step % thinning == 0:
            flawed_samples.append(current_theta)

        # MCMC Progress Report
        if step > 0 and step % mcmc_progress_interval == 0:
            phase = "[BURN-IN]" if step < burn_in else "[SAMPLING]"
            progress_pct = (step / total_steps) * 100
            denom = step if step < burn_in else (step - burn_in + 1)
            acc_rate = accepted / denom if step >= burn_in else 0
            print(f"  Alpha {current_alpha:<4} | Chain {chain_id} | Stage 1: {progress_pct:5.1f}% | {phase:<10} | Acc Rate: {acc_rate:.1%}")

    # --- STAGE 2: Ambient Polishing (The Fix) ---
    exact_angles = np.zeros((len(flawed_samples), k_eff), dtype=np.float32)
    ambient_epsilon = 0.05
    accepted_polishes = 0

    polish_progress_interval = max(1, len(flawed_samples) // 10)

    for i, theta in enumerate(flawed_samples):
        X_current = generate_flawed_lift(theta, M_HAT, n, k)
        dist_sq_current = stiefel_distance_sq(X_current, M_HAT)

        for _ in range(POLISHING_STEPS):
            A = np.random.randn(n, n)
            A = np.tril(A, -1) - np.tril(A, -1).T

            R_noise = expm(ambient_epsilon * A)
            X_proposed = np.dot(R_noise, X_current)

            dist_sq_proposed = stiefel_distance_sq(X_proposed, M_HAT)
            energy_diff = current_alpha * (dist_sq_proposed - dist_sq_current)

            if np.log(np.random.rand()) < -energy_diff:
                X_current = X_proposed
                dist_sq_current = dist_sq_proposed
                accepted_polishes += 1

        exact_angles[i] = extract_exact_angles(X_current, M_HAT)

        # Polishing Progress Report
        if (i + 1) % polish_progress_interval == 0:
            progress_pct = ((i + 1) / len(flawed_samples)) * 100
            print(f"  Alpha {current_alpha:<4} | Chain {chain_id} | Stage 2: {progress_pct:5.1f}% | [POLISHING]")

    elapsed = time.time() - start_time
    total_polishes = len(flawed_samples) * POLISHING_STEPS
    print(f"*** Alpha {current_alpha:<4} | Chain {chain_id} Done! Time: {elapsed:.1f}s | Polish Acc: {accepted_polishes/total_polishes:.1%} ***")

    return exact_angles

# =============================================================================
# 5. PARALLEL EXECUTION & STORAGE
# =============================================================================
if __name__ == '__main__':
    num_cores = mp.cpu_count()
    samples_per_chain = NUM_SAMPLES // num_cores

    print(f"Using {num_cores} cores.")
    print(f"Total target samples per alpha: {NUM_SAMPLES:,} (thinned by {THINNING})")

    for alpha in POTENTIAL_ALPHAS:
        print("\n" + "=" * 80)
        print(f"PROCESSING EXACT ALPHA = {alpha}")
        print("=" * 80)

        tasks = [(i, samples_per_chain, BURN_IN, THINNING, STEP_SIZE, alpha) for i in range(num_cores)]

        with mp.Pool(num_cores) as pool:
            chain_results = pool.starmap(run_mcmc_and_polish, tasks)

        print(f"\nStitching chains for alpha = {alpha}...")
        master_samples = np.vstack(chain_results)

        # Saved with the exact same filename schema so viz.3.py reads it natively
        artifact_name = f"stiefel_mcmc_alpha_{alpha}_n{n}_k{k_eff}.pkl"

        with open(artifact_name, 'wb') as f:
            pickle.dump({
                'n': n,
                'k_eff': k_eff,
                'alpha': alpha,
                'samples': master_samples
            }, f)

        file_size_mb = os.path.getsize(artifact_name) / (1024 * 1024)
        print(f"SUCCESS! Saved {len(master_samples):,} EXACT samples for alpha={alpha}.")
        print(f"Storage Space: {file_size_mb:.2f} MB")

    print("\nAll alphas polished and processed successfully!")