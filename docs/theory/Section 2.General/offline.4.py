import numpy as np
import pickle
import time
import os
import multiprocessing as mp
import torch

# Force PyTorch to use 1 thread per process to prevent CPU locking during multiprocessing
torch.set_num_threads(1)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
n = 400
k = 100
k_eff = min(k, n - k)

# 8 values linearly spaced from 0.1 to 10.0
POTENTIAL_ALPHAS = np.linspace(0.1, 10.0, 8)

# MCMC Settings
NUM_SAMPLES = 10_000      # Reduced because MALA is much more sample-efficient
BURN_IN = 50000
THINNING = 1               # MALA de-correlates fast enough that we can often keep every sample
INITIAL_STEP_SIZE = 1e-5   # Langevin step sizes (tau) are typically smaller than RWMH standard deviations
NUM_CHAINS = 1

# The theoretical optimal acceptance rate for MALA is ~0.574
#target_accept = 0.574
target_accept = 0.574

print(f"Initializing MALA Offline Precomputation for Stiefel V({n}, {k})...")

# =============================================================================
# 2. THE TARGET DENSITY & GRADIENT (PyTorch AutoDiff)
# =============================================================================
def get_log_p_and_grad(theta_np, current_alpha):
    """
    Computes the log-probability and its exact gradient using PyTorch.
    Returns (log_p, gradient) or (-np.inf, None) if out of bounds.
    """
    # 1. Reject invalid states (Boundary checks)
    if theta_np[0] >= np.pi or theta_np[-1] <= 0.0:
        return -np.inf, None
    if k_eff > 1 and theta_np[0] + theta_np[1] >= np.pi:
        return -np.inf, None
    if np.any(np.diff(theta_np) >= 0):
        return -np.inf, None

    # 2. Convert to Torch tensor with gradient tracking (use float64 for precision)
    theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)

    short_power = n - 2 * k_eff
    log_weyl = 0.0

    # Boundary interaction root
    if short_power > 0:
        log_weyl = log_weyl + short_power * torch.sum(torch.log(torch.abs(torch.sin(theta))))

    # Internal frame interaction roots
    if k_eff > 1:
        diffs = theta.unsqueeze(1) - theta.unsqueeze(0)
        sums = theta.unsqueeze(1) + theta.unsqueeze(0)
        idx_row, idx_col = torch.triu_indices(k_eff, k_eff, offset=1)

        log_weyl = log_weyl + torch.sum(torch.log(torch.abs(torch.sin(diffs[idx_row, idx_col]))))
        log_weyl = log_weyl + torch.sum(torch.log(torch.abs(torch.sin(sums[idx_row, idx_col]))))

    log_potential = -current_alpha * torch.sum(theta**2)

    log_p = log_weyl + log_potential

    # 3. Compute gradients automatically
    log_p.backward()

    return log_p.item(), theta.grad.numpy()

# =============================================================================
# 3. MALA ENGINE (Langevin Dynamics + Metropolis Hastings)
# =============================================================================
def run_mala(chain_id, num_samples, burn_in, thinning, tau, current_alpha):
    """Runs a single MALA chain."""
    # Initialize safely inside the Weyl chamber
    current_theta = np.linspace(np.pi/2 - 0.05, 0.05, k_eff, dtype=np.float64)
    current_log_p, current_grad = get_log_p_and_grad(current_theta, current_alpha)

    samples = np.zeros((num_samples // thinning, k_eff), dtype=np.float32)

    burn_in_accepted = 0
    sampling_accepted = 0
    total_steps = num_samples + burn_in
    save_idx = 0

    np.random.seed(int(os.getpid() + chain_id + current_alpha * 1000))
    start_time = time.time()
    progress_interval = max(1, total_steps // 10)

    for step in range(total_steps):
        # 1. Propose new state using Langevin dynamics: x' = x + tau * grad + sqrt(2 * tau) * noise
        noise = np.random.normal(0, 1, k_eff)
        mean_curr = current_theta + tau * current_grad
        proposed_theta = mean_curr + np.sqrt(2 * tau) * noise

        # 2. Evaluate target density and gradient at proposed state
        proposed_log_p, proposed_grad = get_log_p_and_grad(proposed_theta, current_alpha)

        accepted_this_step = 0

        if proposed_log_p != -np.inf:
            # 3. Calculate transition probabilities q(x'|x) and q(x|x')
            mean_prop = proposed_theta + tau * proposed_grad

            log_q_curr_to_prop = -np.sum((proposed_theta - mean_curr)**2) / (4 * tau)
            log_q_prop_to_curr = -np.sum((current_theta - mean_prop)**2) / (4 * tau)

            # 4. Metropolis acceptance ratio
            log_accept_ratio = (proposed_log_p - current_log_p) + (log_q_prop_to_curr - log_q_curr_to_prop)

            if np.log(np.random.rand()) < log_accept_ratio:
                current_theta = proposed_theta
                current_log_p = proposed_log_p
                current_grad = proposed_grad
                accepted_this_step = 1

        if step < burn_in:
            burn_in_accepted += accepted_this_step
            # Robbins-Monro step size adaptation
            gamma = 1.0 / (step + 50)**0.6
            tau *= np.exp(gamma * (accepted_this_step - target_accept))
            # Clip tau to prevent numerical explosion during early burn-in
            tau = np.clip(tau, 1e-8, 1e-2)
        else:
            sampling_accepted += accepted_this_step

        if step >= burn_in and step % thinning == 0:
            samples[save_idx] = current_theta
            save_idx += 1

        if step > 0 and step % progress_interval == 0:
            if step < burn_in:
                acc_rate = burn_in_accepted / step
                phase = "[BURN-IN]"
            else:
                acc_rate = sampling_accepted / (step - burn_in + 1)
                phase = "[SAMPLING]"

            progress_pct = (step / total_steps) * 100
            print(f"  Alpha {current_alpha:<7.4f} | {progress_pct:5.1f}% | {phase:<10} | Acc Rate: {acc_rate:.1%} | Tau: {tau:.2e}")

    elapsed = time.time() - start_time
    final_acc = sampling_accepted / num_samples
    print(f"*** Alpha {current_alpha:<7.4f} Completed! Time: {elapsed:.1f}s | Final Acc: {final_acc:.1%} | Final Tau: {tau:.2e} ***")

    return samples

# =============================================================================
# 4. WORKER FUNCTION FOR PARALLEL MAPPING
# =============================================================================
def process_alpha(alpha):
    samples_per_chain = NUM_SAMPLES // NUM_CHAINS
    chain_results = []

    for i in range(NUM_CHAINS):
        samples = run_mala(i, samples_per_chain, BURN_IN, THINNING, INITIAL_STEP_SIZE, alpha)
        chain_results.append(samples)

    master_samples = np.vstack(chain_results)
    artifact_name = f"stiefel_mala_alpha_{alpha:.4f}_n{n}_k{k_eff}.pkl"

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
    # Reserve a core for the OS if you have many cores, otherwise use all
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