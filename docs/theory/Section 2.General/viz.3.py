import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from scipy.stats import skewnorm

# =============================================================================
# 1. FIXED PARAMETERS
# =============================================================================
POTENTIAL_ALPHAS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# alpha=lambda/(delta^2)
n = 40
k = 10
k_eff = min(k, n - k)
BINS = 80

# =============================================================================
# 2. DATA LOADING
# =============================================================================
valid_alphas = []
all_samples = {}

print(f"Scanning for MCMC files (n={n}, k_eff={k_eff})...")

for a in POTENTIAL_ALPHAS:
    filename = f"stiefel_mcmc_alpha_{a}_n{n}_k{k_eff}.pkl"
    if os.path.exists(filename):
        print(f"  -> Found {filename}")
        valid_alphas.append(a)

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            all_samples[a] = data['samples']

if not valid_alphas:
    print(f"Error: No .pkl files found. Run offline.3.py first!")
    exit()

print("Data loaded successfully. Launching interactive MCMC Diagnostics...")

# =============================================================================
# 3. INTERACTIVE DUAL-PANEL VISUALIZATION
# =============================================================================
# Global UI State
state = {
    'alpha': valid_alphas[0],
    'angle_idx': 0
}

# --- UI Setup ---
plt.style.use('seaborn-v0_8-darkgrid')

fig, (ax_trace, ax_hist) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [1, 1.5]})
plt.subplots_adjust(left=0.20, bottom=0.08, top=0.92, hspace=0.35)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, k_eff))

def draw_plots():
    ax_trace.clear()
    ax_hist.clear()

    alpha = state['alpha']
    angle_idx = state['angle_idx']

    max_trace_points = min(10000, len(all_samples[alpha]))
    data_trace = all_samples[alpha][:max_trace_points, angle_idx]

    data_full = all_samples[alpha][:, angle_idx]

    # Adaptive Bounding
    p_low = np.percentile(data_full, 0.1)
    p_high = np.percentile(data_full, 99.9)
    margin = max((p_high - p_low) * 0.1, 0.02)
    plot_min = max(0.0, p_low - margin)
    plot_max = min(np.pi, p_high + margin)

    # ---------------------------------------------------------
    # PANEL 1: THE TRACE PLOT (MCMC Diagnostic)
    # ---------------------------------------------------------
    ax_trace.plot(data_trace, color=colors[angle_idx], alpha=0.8, linewidth=0.5)
    ax_trace.set_title(f"MCMC Trace Plot: $\\theta_{{{angle_idx+1}}}$ (First {max_trace_points:,} saved steps)", fontsize=13, fontweight='bold')
    ax_trace.set_ylabel('Angle (Radians)', fontsize=11)
    ax_trace.set_xlabel('Thinned Sample Index', fontsize=11)
    ax_trace.set_ylim(plot_min, plot_max)

    mean_val = np.mean(data_trace)
    ax_trace.axhline(mean_val, color='black', linestyle='--', alpha=0.5, label=f"Mean: {mean_val:.3f}")
    ax_trace.legend(loc='upper right')

    # ---------------------------------------------------------
    # PANEL 2: THE HISTOGRAM & SKEW-NORMAL FIT
    # ---------------------------------------------------------
    custom_bins = np.linspace(plot_min, plot_max, BINS)

    ax_hist.hist(data_full, bins=custom_bins, density=True,
                 color=colors[angle_idx], alpha=0.7, edgecolor='white', linewidth=0.5,
                 label=f'MCMC Samples (N={len(data_full):,})')

    # Parametric Fit: Skew-Normal Distribution
    # Subsample for faster fitting to keep the UI snappy when clicking buttons
    fit_data = data_full if len(data_full) < 5000 else np.random.choice(data_full, 5000, replace=False)

    # Fit the skewness (ae), location (loce), and scale (scalee)
    ae, loce, scalee = skewnorm.fit(fit_data)

    x_smooth = np.linspace(plot_min, plot_max, 500)
    y_smooth = skewnorm.pdf(x_smooth, ae, loce, scalee)

    ax_hist.plot(x_smooth, y_smooth, color='crimson', linewidth=2.5,
                 label=f'Theoretical PDF (Skew-Normal, $\\alpha_s$={ae:.2f})')

    ax_hist.set_title(f"Marginal Density of $\\theta_{{{angle_idx+1}}}$ ($\\alpha={alpha}$)", fontsize=14, fontweight='bold')
    ax_hist.set_xlim(plot_min, plot_max)
    ax_hist.set_xlabel('Angle (Radians)', fontsize=11)
    ax_hist.set_ylabel('Probability Density', fontsize=11)
    ax_hist.legend(loc='upper left')

    fig.canvas.draw_idle()

# --- Interactive Menus ---

# 1. Alpha Menu (Top Left)
ax_menu_alpha = plt.axes([0.02, 0.75, 0.12, 0.15])
ax_menu_alpha.set_facecolor('whitesmoke')
ax_menu_alpha.set_title("Select Alpha", fontweight='bold')
alpha_labels = [str(a) for a in valid_alphas]
radio_alpha = RadioButtons(ax_menu_alpha, alpha_labels, active=0)

def on_alpha_click(label):
    state['alpha'] = float(label)
    draw_plots()

radio_alpha.on_clicked(on_alpha_click)

# 2. Angle Menu (Bottom Left)
ax_menu_angle = plt.axes([0.02, 0.05, 0.12, 0.65])
ax_menu_angle.set_facecolor('whitesmoke')
ax_menu_angle.set_title("Select Angle", fontweight='bold')
angle_labels = [f"Theta {i+1}" for i in range(k_eff)]
radio_angle = RadioButtons(ax_menu_angle, angle_labels, active=0)

def on_angle_click(label):
    state['angle_idx'] = int(label.split()[1]) - 1
    draw_plots()

radio_angle.on_clicked(on_angle_click)

# Draw initial state
draw_plots()
plt.show()