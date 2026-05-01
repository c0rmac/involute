import numpy as np
import pickle
import time
import os
import multiprocessing as mp
from numba import njit

# =============================================================================
# 1. PARAMETERS
# =============================================================================
n = 1000
k = 100
# Enforcing the geometric rank based on the Stiefel constraints
k_eff = min(k, n - k)

# 8 values linearly spaced from 0.1 to 10.0 representing alpha (lambda/delta^2)
POTENTIAL_ALPHAS = np.linspace(0.1, 10.0, 8)

# HMC Settings
NUM_SAMPLES = 100_000
BURN_IN = 25_000
THINNING = 1
NUM_LEAPFROG_STEPS = 5
INITIAL_STEP_SIZE = 1e-4
NUM_CHAINS = 1

# Theoretical optimal acceptance rate for HMC
TARGET_ACCEPT = 0.65

# --- DIAGONAL MASS MATRIX (PRECONDITIONING) ---
# We assign larger mass (1.0) to theta_1 and smaller mass (0.05) to theta_100.
# A smaller mass allows the "lighter" angles near the boundary to take larger steps.
MASS_DIAG = np.linspace(1.0, 0.05, k_eff)
INV_MASS = 1.0 / MASS_DIAG

print(f"Initializing Analytical Dyson Drift HMC with Preconditioning for Stiefel V({n}, {k})...")

# =============================================================================
# 2. THE TARGET DENSITY & DYSON DRIFT (Compiled to C via Numba)
# =============================================================================
@njit
def get_log_p_and_dyson_drift(theta, current_alpha, n_val, k_val):
    """
    Computes the log-probability and the exact analytical Dyson Drift.
    """
    # 1. Reject invalid states (Boundary checks: pi > theta_1 > ... > theta_k > 0)
    if theta[0] >= np.pi or theta[-1] <= 0.0:
        return -np.inf, np.zeros_like(theta)

    # Enforce strictly decreasing principal angles
    for i in range(k_val - 1):
        if theta[i + 1] >= theta[i]:
            return -np.inf, np.zeros_like(theta)

    log_p = 0.0
    grad = np.zeros(k_val, dtype=np.float64)
    boundary_multiplicity = n_val - 2 * k_val

    # 2. Compute Analytical Gradients via the Generalized Dyson Drift
    for i in range(k_val):
        # A. Consensus Drift
        log_p -= current_alpha * theta[i]**2
        grad[i] -= 2.0 * current_alpha * theta[i]

        # B. Boundary Interaction
        if boundary_multiplicity > 0:
            log_p += boundary_multiplicity * np.log(np.sin(theta[i]))
            grad[i] += boundary_multiplicity / np.tan(theta[i])

        # C. Internal Frame Interaction (Dyson Repulsion)
        for j in range(i + 1, k_val):
            diff = theta[i] - theta[j]
            sum_ij = theta[i] + theta[j]

            log_p += np.log(np.sin(diff)) + np.log(np.sin(sum_ij))

            cot_diff = 1.0 / np.tan(diff)
            cot_sum = 1.0 / np.tan(sum_ij)

            # Accumulate repulsive forces
            grad[i] += cot_diff + cot_sum
            grad[j] += -cot_diff + cot_sum

    return log_p, grad

# =============================================================================
# 3. HIGH-PERFORMANCE HMC ENGINE (Compiled)
# =============================================================================
@njit
def hmc_core(num_samples, burn_in, thinning, epsilon, L, current_alpha, n_val, k_val, mass_diag, inv_mass, seed):
    """
    The main HMC loop, fully compiled via Numba.
    Now utilizes the exact Dyson Drift and a Diagonal Mass Matrix.
    """
    np.random.seed(seed)

    # Initialize safely inside the fundamental Weyl chamber
    current_theta = np.linspace(np.pi/2 - 0.05, 0.05, k_val)
    current_log_p, current_grad = get_log_p_and_dyson_drift(current_theta, current_alpha, n_val, k_val)

    samples = np.zeros((num_samples // thinning, k_val), dtype=np.float32)

    burn_in_accepted = 0
    sampling_accepted = 0
    total_steps = num_samples + burn_in
    save_idx = 0

    progress_interval = max(1, total_steps // 10)

    for step in range(total_steps):
        # 1. Sample random momentum based on Mass Matrix (p ~ N(0, M))
        # Note: Numba requires a loop for array-based scales in np.random.normal
        p_curr = np.zeros(k_val, dtype=np.float64)
        for dim in range(k_val):
            p_curr[dim] = np.random.normal(0.0, np.sqrt(mass_diag[dim]))

        # Kinetic Energy: K(p) = 0.5 * p^T M^-1 p
        H_curr = -current_log_p + 0.5 * np.sum(p_curr**2 * inv_mass)

        theta_prop = current_theta.copy()
        p_prop = p_curr.copy()
        grad_prop = current_grad.copy()
        valid_trajectory = True

        # --- LEAPFROG INTEGRATOR ---
        # Half step for momentum (force is independent of mass)
        p_prop += 0.5 * epsilon * grad_prop

        for i in range(L):
            # Full step for position (velocity = M^-1 p)
            theta_prop += epsilon * (p_prop * inv_mass)

            prop_log_p, grad_prop = get_log_p_and_dyson_drift(theta_prop, current_alpha, n_val, k_val)

            if prop_log_p == -np.inf:
                valid_trajectory = False
                break

            if i != L - 1:
                p_prop += epsilon * grad_prop

        accepted_this_step = 0

        if valid_trajectory:
            # Final half step for momentum
            p_prop += 0.5 * epsilon * grad_prop

            H_prop = -prop_log_p + 0.5 * np.sum(p_prop**2 * inv_mass)
            log_accept_ratio = H_curr - H_prop

            if np.log(np.random.rand()) < log_accept_ratio:
                current_theta = theta_prop
                current_log_p = prop_log_p
                current_grad = grad_prop
                accepted_this_step = 1

        # 2. Adapt Epsilon
        if step < burn_in:
            burn_in_accepted += accepted_this_step
            gamma = 1.0 / (step + 50)**0.6
            epsilon *= np.exp(gamma * (accepted_this_step - TARGET_ACCEPT))
            if epsilon < 1e-6: epsilon = 1e-6
            if epsilon > 1e-2: epsilon = 1e-2
        else:
            sampling_accepted += accepted_this_step

        # 3. Save Thinned Samples
        if step >= burn_in and step % thinning == 0:
            for dim in range(k_val):
                samples[save_idx, dim] = np.float32(current_theta[dim])
            save_idx += 1

        # ---------------------------------------------------------------------
        # PROGRESS LOGGING (Numba Safe)
        # ---------------------------------------------------------------------
        if step > 0 and step % progress_interval == 0:
            if step < burn_in:
                acc_rate = burn_in_accepted / step
                phase = "[BURN-IN] "
            else:
                acc_rate = sampling_accepted / (step - burn_in + 1)
                phase = "[SAMPLING]"

            progress_pct = (step / total_steps) * 100.0

            print("  Alpha", round(current_alpha, 4),
                  "|", round(progress_pct, 1), "%",
                  "|", phase,
                  "| Acc Rate:", round(acc_rate, 3),
                  "| Eps:", epsilon)

    return samples, epsilon, sampling_accepted

def run_hmc_wrapper(chain_id, num_samples, burn_in, thinning, epsilon, L, current_alpha):
    seed = int(os.getpid() + chain_id + current_alpha * 1000)
    start_time = time.time()

    samples, final_eps, sampling_accepted = hmc_core(
        num_samples, burn_in, thinning, epsilon, L, current_alpha, n, k_eff,
        MASS_DIAG, INV_MASS, seed
    )

    elapsed = time.time() - start_time
    final_acc = sampling_accepted / num_samples
    print(f"*** Alpha {current_alpha:<7.4f} Completed! Time: {elapsed:.1f}s | Final Acc: {final_acc:.1%} | Final Eps: {final_eps:.2e} ***")

    return samples

# =============================================================================
# 4. WORKER FUNCTION FOR PARALLEL MAPPING
# =============================================================================
def process_alpha(alpha):
    samples_per_chain = NUM_SAMPLES // NUM_CHAINS
    chain_results = []

    for i in range(NUM_CHAINS):
        samples = run_hmc_wrapper(i, samples_per_chain, BURN_IN, THINNING, INITIAL_STEP_SIZE, NUM_LEAPFROG_STEPS, alpha)
        chain_results.append(samples)

    master_samples = np.vstack(chain_results)
    artifact_name = f"stiefel_hmc_alpha_{alpha:.4f}_n{n}_k{k_eff}.pkl"

    with open(artifact_name, 'wb') as f:
        pickle.dump({
            'n': n,
            'k_eff': k_eff,
            'alpha': alpha,
            'samples': master_samples
        }, f)

    return alpha, len(master_samples), artifact_name

# =============================================================================
# 5. PARALLEL EXECUTION OVER ALPHAS
# =============================================================================
if __name__ == '__main__':
    num_cores = mp.cpu_count()

    print(f"Using {num_cores} cores to process {len(POTENTIAL_ALPHAS)} alphas.")
    print(f"Target samples per alpha: {NUM_SAMPLES:,} (thinned by {THINNING})")
    print("=" * 80)

    with mp.Pool(num_cores) as pool:
        results = pool.map(process_alpha, POTENTIAL_ALPHAS)

    print("\n" + "=" * 80)
    print("ALL ALPHAS PROCESSED SUCCESSFULLY!")
    print("=" * 80)

    for res_alpha, num_saved, filename in results:
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"Alpha {res_alpha:<7.4f} -> Saved {num_saved:,} samples | File: {filename} ({file_size_mb:.2f} MB)")