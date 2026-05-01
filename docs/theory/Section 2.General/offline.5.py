import numpy as np
import pickle
import time
import os
import multiprocessing as mp
from numba import njit

# =============================================================================
# 1. PARAMETERS
# =============================================================================
n = 400
k = 100
k_eff = min(k, n - k)

# 8 values linearly spaced from 0.1 to 10.0
POTENTIAL_ALPHAS = np.linspace(0.1, 10.0, 8)

# HMC Settings
NUM_SAMPLES = 100_000
BURN_IN = 1000
THINNING = 1
NUM_LEAPFROG_STEPS = 40
INITIAL_STEP_SIZE = 1e-4
NUM_CHAINS = 1

# Theoretical optimal acceptance rate for HMC
TARGET_ACCEPT = 0.65

print(f"Initializing Ultra-Fast HMC (Numba JIT) for Stiefel V({n}, {k})...")

# =============================================================================
# 2. THE TARGET DENSITY & EXACT ANALYTICAL GRADIENT (Compiled to C via Numba)
# =============================================================================
@njit
def get_log_p_and_grad(theta, current_alpha, n_val, k_val):
    """
    Computes the log-probability and analytical gradient simultaneously.
    Using @njit forces this into compiled C code, making it incredibly fast.
    """
    # 1. Reject invalid states (Boundary checks)
    if theta[0] >= np.pi or theta[-1] <= 0.0:
        return -np.inf, np.zeros_like(theta)
    if k_val > 1 and theta[0] + theta[1] >= np.pi:
        return -np.inf, np.zeros_like(theta)

    # Check if strictly decreasing
    for i in range(k_val - 1):
        if theta[i + 1] >= theta[i]:
            return -np.inf, np.zeros_like(theta)

    log_p = 0.0
    grad = np.zeros(k_val, dtype=np.float64)
    short_power = n_val - 2 * k_val

    # 2. Compute Analytical Gradients
    for i in range(k_val):
        # Potential term
        log_p -= current_alpha * theta[i]**2
        grad[i] -= 2.0 * current_alpha * theta[i]

        # Boundary interaction term
        if short_power > 0:
            log_p += short_power * np.log(np.abs(np.sin(theta[i])))
            grad[i] += short_power / np.tan(theta[i])

        # Internal frame interaction roots (Pairs)
        for j in range(i + 1, k_val):
            diff = theta[i] - theta[j]
            sum_ij = theta[i] + theta[j]

            log_p += np.log(np.abs(np.sin(diff)))
            log_p += np.log(np.abs(np.sin(sum_ij)))

            cot_diff = 1.0 / np.tan(diff)
            cot_sum = 1.0 / np.tan(sum_ij)

            # Accumulate gradients for both i and j components
            grad[i] += cot_diff + cot_sum
            grad[j] += -cot_diff + cot_sum

    return log_p, grad

# =============================================================================
# 3. HIGH-PERFORMANCE HMC ENGINE (Compiled)
# =============================================================================
# =============================================================================
# 3. HIGH-PERFORMANCE HMC ENGINE (Compiled)
# =============================================================================
@njit
def hmc_core(num_samples, burn_in, thinning, epsilon, L, current_alpha, n_val, k_val, seed):
    """
    The main HMC loop, fully compiled via Numba.
    Now includes safe progress logging without f-string format specifiers.
    """
    np.random.seed(seed)

    # Initialize safely inside the Weyl chamber
    current_theta = np.linspace(np.pi/2 - 0.05, 0.05, k_val)
    current_log_p, current_grad = get_log_p_and_grad(current_theta, current_alpha, n_val, k_val)

    samples = np.zeros((num_samples // thinning, k_val), dtype=np.float32)

    burn_in_accepted = 0
    sampling_accepted = 0
    total_steps = num_samples + burn_in
    save_idx = 0

    # Calculate when to print progress logs
    progress_interval = max(1, total_steps // 10)

    for step in range(total_steps):
        # 1. Sample random momentum
        p_curr = np.random.normal(0.0, 1.0, k_val)
        H_curr = -current_log_p + 0.5 * np.sum(p_curr**2)

        theta_prop = current_theta.copy()
        p_prop = p_curr.copy()
        grad_prop = current_grad.copy()
        valid_trajectory = True

        # --- LEAPFROG INTEGRATOR ---
        p_prop += 0.5 * epsilon * grad_prop

        for i in range(L):
            theta_prop += epsilon * p_prop
            prop_log_p, grad_prop = get_log_p_and_grad(theta_prop, current_alpha, n_val, k_val)

            if prop_log_p == -np.inf:
                valid_trajectory = False
                break

            if i != L - 1:
                p_prop += epsilon * grad_prop

        accepted_this_step = 0

        if valid_trajectory:
            p_prop += 0.5 * epsilon * grad_prop
            H_prop = -prop_log_p + 0.5 * np.sum(p_prop**2)

            log_accept_ratio = H_curr - H_prop

            if np.log(np.random.rand()) < log_accept_ratio:
                current_theta = theta_prop
                current_log_p = prop_log_p
                current_grad = grad_prop
                accepted_this_step = 1

        # Adapt Epsilon
        if step < burn_in:
            burn_in_accepted += accepted_this_step
            gamma = 1.0 / (step + 50)**0.6
            epsilon *= np.exp(gamma * (accepted_this_step - TARGET_ACCEPT))
            if epsilon < 1e-6: epsilon = 1e-6
            if epsilon > 1e-2: epsilon = 1e-2
        else:
            sampling_accepted += accepted_this_step

        # Save Samples
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

            # Numba-safe printing: No format specifiers, using round() instead
            print("  Alpha", round(current_alpha, 4),
                  "|", round(progress_pct, 1), "%",
                  "|", phase,
                  "| Acc Rate:", round(acc_rate, 3),
                  "| Eps:", epsilon)

    return samples, epsilon, sampling_accepted

def run_hmc_wrapper(chain_id, num_samples, burn_in, thinning, epsilon, L, current_alpha):
    """Python wrapper to manage JIT compilation prints and time tracking."""
    seed = int(os.getpid() + chain_id + current_alpha * 1000)
    start_time = time.time()

    # Note: The first alpha will take slightly longer due to Numba JIT compilation
    samples, final_eps, sampling_accepted = hmc_core(
        num_samples, burn_in, thinning, epsilon, L, current_alpha, n, k_eff, seed
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

    # Note on the first run: Numba has to compile the C-code. The first alpha might take
    # 2-3 seconds longer to initialize, but the rest will be lightning fast.
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_alpha, POTENTIAL_ALPHAS)

    print("\n" + "=" * 80)
    print("ALL ALPHAS PROCESSED SUCCESSFULLY VIA FAST JIT HMC!")
    print("=" * 80)

    for res_alpha, num_saved, filename in results:
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"Alpha {res_alpha:<7.4f} -> Saved {num_saved:,} samples | File: {filename} ({file_size_mb:.2f} MB)")