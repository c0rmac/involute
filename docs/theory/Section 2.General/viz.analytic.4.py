import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import time

# =============================================================================
# 1. FIXED PARAMETERS
# =============================================================================
n = 40  # Must match offline.4.py
k = 20
k_eff = min(k, n - k)
BINS = 60

# The alphas we want to be able to instantly simulate in the UI
UI_ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 100.0]

# =============================================================================
# 2. DATA LOADING (Only the Universal Base)
# =============================================================================
base_file = f"base_algebraic_z_n{n}_k{k_eff}.pkl"
base_z_samples = None

print(f"Scanning for Universal Base: {base_file}")
if os.path.exists(base_file):
    with open(base_file, 'rb') as f:
        base_z_samples = pickle.load(f)
    print(f"  -> Success! Loaded {len(base_z_samples):,} base candidates.")
else:
    print(f"Error: Could not find {base_file}. Run offline.4.py first!")
    exit()

# =============================================================================
# 3. REAL-TIME IMPORTANCE SAMPLING MATH
# =============================================================================
def compute_importance_weights(theta_batch):
    """Calculates exact Importance Sampling weights instantly."""
    N_samples = theta_batch.shape[0]
    log_w = np.zeros(N_samples)

    valid_mask = theta_batch[:, 0] < np.pi
    invalid_mask = ~valid_mask

    valid_theta = theta_batch[valid_mask]
    log_w_valid = np.zeros(len(valid_theta))

    short_power = n - 2 * k_eff
    eps = 1e-12

    if short_power > 0:
        log_w_valid += short_power * np.sum(
            np.log(np.sin(valid_theta) + eps) - np.log(valid_theta + eps), axis=1
        )

    if k_eff > 1:
        for i in range(k_eff):
            for j in range(i + 1, k_eff):
                diffs = valid_theta[:, i] - valid_theta[:, j]
                sums = valid_theta[:, i] + valid_theta[:, j]

                log_w_valid += np.log(np.sin(diffs) + eps) - np.log(diffs + eps)
                log_w_valid += np.log(np.sin(sums) + eps) - np.log(sums + eps)

    log_w[valid_mask] = log_w_valid
    log_w[invalid_mask] = -np.inf

    weights = np.exp(log_w)

    sum_weights = np.sum(weights)
    if sum_weights > 0:
        weights = weights / sum_weights

    return weights

dynamic_cache = {}

def get_dynamic_data(alpha):
    """Scales the base samples and computes weights on the fly."""
    if alpha not in dynamic_cache:
        t0 = time.time()
        theta = base_z_samples / np.sqrt(alpha)
        weights = compute_importance_weights(theta)

        dynamic_cache[alpha] = (theta, weights)
        print(f"Dynamic Alpha {alpha} generated in {time.time()-t0:.3f}s")

    return dynamic_cache[alpha]

# =============================================================================
# 4. INTERACTIVE DUAL-PANEL VISUALIZATION
# =============================================================================
state = {
    'alpha': UI_ALPHAS[0],
    'angle_idx': 0
}

plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax_trace, ax_hist) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [1, 1.5]})
plt.subplots_adjust(left=0.20, bottom=0.08, top=0.92, hspace=0.35)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, k_eff))

def draw_plots():
    ax_trace.clear()
    ax_hist.clear()

    alpha = state['alpha']
    angle_idx = state['angle_idx']
    c = colors[angle_idx]

    # Dynamically generate or fetch the weighted data
    theta_dynamic, weights = get_dynamic_data(alpha)
    data_weight = theta_dynamic[:, angle_idx]

    # --- FIX: Filter out invalid (0-weight) samples for the histogram ---
    valid_mask = weights > 0
    valid_data = data_weight[valid_mask]
    valid_weights = weights[valid_mask]

    # Calculate bounds ONLY on valid data
    if len(valid_data) > 0:
        p_low = np.percentile(valid_data, 0.1)
        p_high = np.percentile(valid_data, 99.9)
    else:
        p_low, p_high = 0.0, np.pi # Fallback if Alpha is so low everything is rejected

    margin = max((p_high - p_low) * 0.1, 0.02)
    hist_min = max(0.0, p_low - margin)
    hist_max = min(np.pi, p_high + margin)

    # Safety check to absolutely guarantee monotonic bins
    if hist_min >= hist_max:
        hist_max = hist_min + 0.1

    # ---------------------------------------------------------
    # PANEL 1: TRACE PLOT (Scaled Base Samples)
    # ---------------------------------------------------------
    max_trace_points = min(10000, len(data_weight))
    data_trace = data_weight[:max_trace_points]

    ax_trace.plot(data_trace, color=c, alpha=0.8, linewidth=0.5)
    ax_trace.set_title(f"Scaled Base Trace: $\\theta_{{{angle_idx+1}}}$ (First {max_trace_points:,} steps)", fontsize=13, fontweight='bold')

    # Draw the Pi boundary line so you can see what gets rejected!
    ax_trace.axhline(np.pi, color='red', linestyle='-', alpha=0.3, label="$\pi$ Boundary (Rejection Wall)")

    mean_val = np.average(valid_data, weights=valid_weights) if len(valid_data) > 0 else 0
    ax_trace.axhline(mean_val, color='black', linestyle='--', alpha=0.5, label=f"Weighted Mean: {mean_val:.3f}")
    ax_trace.legend(loc='upper right')

    ax_trace.set_ylabel('Angle (Radians)', fontsize=11)
    ax_trace.set_xlabel('Thinned Sample Index', fontsize=11)

    # Let the trace plot expand dynamically so you can see out-of-bounds samples
    trace_max = max(np.pi + 0.2, np.max(data_trace) + 0.1)
    ax_trace.set_ylim(0, trace_max)

    # ---------------------------------------------------------
    # PANEL 2: IMPORTANCE SAMPLING HISTOGRAM
    # ---------------------------------------------------------
    custom_bins = np.linspace(hist_min, hist_max, BINS)
    n_eff = 1.0 / np.sum(weights**2) if np.sum(weights) > 0 else 0

    if len(valid_data) > 0:
        ax_hist.hist(valid_data, bins=custom_bins, weights=valid_weights, density=True,
                     color=c, alpha=0.7, edgecolor='white', linewidth=0.5,
                     label=f'Importance Sampling (N_eff $\\approx$ {n_eff:,.0f})')
    else:
        ax_hist.text(0.5, 0.5, "Alpha too low. All samples exceeded Pi.",
                     ha='center', va='center', transform=ax_hist.transAxes)

    ax_hist.set_title(f"Marginal Density of $\\theta_{{{angle_idx+1}}}$ ($\\alpha={alpha}$)", fontsize=14, fontweight='bold')
    ax_hist.set_xlim(hist_min, hist_max)
    ax_hist.set_xlabel('Angle (Radians)', fontsize=11)
    ax_hist.set_ylabel('Probability Density', fontsize=11)
    ax_hist.legend(loc='upper left', fontsize=11)

    fig.canvas.draw_idle()

# --- Interactive Menus ---
ax_menu_alpha = plt.axes([0.02, 0.65, 0.12, 0.25])
ax_menu_alpha.set_facecolor('whitesmoke')
ax_menu_alpha.set_title("Select Alpha", fontweight='bold')
alpha_labels = [str(a) for a in UI_ALPHAS]
radio_alpha = RadioButtons(ax_menu_alpha, alpha_labels, active=0)

def on_alpha_click(label):
    state['alpha'] = float(label)
    draw_plots()
radio_alpha.on_clicked(on_alpha_click)

ax_menu_angle = plt.axes([0.02, 0.05, 0.12, 0.55])
ax_menu_angle.set_facecolor('whitesmoke')
ax_menu_angle.set_title("Select Angle", fontweight='bold')
angle_labels = [f"Theta {i+1}" for i in range(k_eff)]
radio_angle = RadioButtons(ax_menu_angle, angle_labels, active=0)

def on_angle_click(label):
    state['angle_idx'] = int(label.split()[1]) - 1
    draw_plots()
radio_angle.on_clicked(on_angle_click)

draw_plots()
plt.show()