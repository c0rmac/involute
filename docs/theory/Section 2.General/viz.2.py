import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# 1. FIXED PARAMETERS
# =============================================================================
POTENTIAL_ALPHAS = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

n = 20
k = 10
M_local = 8
k_eff = min(k, n - k)
NUM_SAMPLES = 200000
BINS = 80

# =============================================================================
# 2. CONTINUOUS SPLINE SAMPLER
# =============================================================================
def sample_sequential_tree(node, k_eff):
    final_angles = []
    curr_node = node

    for d in range(k_eff):
        if curr_node is None or 'vals' not in curr_node or len(curr_node['vals']) == 0:
            last_val = final_angles[-1] if final_angles else 0.1
            while len(final_angles) < k_eff:
                last_val *= 0.5
                final_angles.append(last_val)
            break

        vals = curr_node['vals']
        cdf = curr_node['cdf']

        u = np.random.rand()

        u_nodes = np.concatenate(([0.0], cdf))
        t_nodes = np.concatenate((vals, [0.0]))

        u_nodes = u_nodes[::-1]
        t_nodes = t_nodes[::-1]

        unique_idx = np.unique(u_nodes, return_index=True)[1]
        u_nodes = u_nodes[unique_idx]
        t_nodes = t_nodes[unique_idx]

        if len(u_nodes) > 1:
            interpolator = PchipInterpolator(u_nodes, t_nodes)
            angle = float(interpolator(u))
        else:
            angle = float(t_nodes[0])

        angle = max(0.0, min(np.pi/2, angle))
        final_angles.append(angle)

        idx = np.searchsorted(cdf, u)
        idx = min(idx, len(vals) - 1)

        if curr_node['children']:
            target_key = vals[idx]
            if target_key in curr_node['children']:
                curr_node = curr_node['children'][target_key]
            else:
                dict_keys = list(curr_node['children'].keys())
                best_key = min(dict_keys, key=lambda x: abs(x - target_key))
                curr_node = curr_node['children'][best_key]
        else:
            curr_node = None

    return np.array(final_angles[:k_eff], dtype=np.float32)

def generate_ensemble(alpha, num_samples):
    filename = f"stiefel_seq_alpha_{alpha:.2f}_n{n}_k{k_eff}_M{M_local}.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    tree = data['sparse_tree']
    print(f"  -> Sampling {num_samples} particles for Alpha = {alpha}...")

    samples = np.zeros((num_samples, k_eff))
    for i in range(num_samples):
        samples[i] = sample_sequential_tree(tree, k_eff)

    return samples

# =============================================================================
# 3. UPFRONT GENERATION & DATA LOADING
# =============================================================================
# Scan for available files
valid_alphas = []
for a in POTENTIAL_ALPHAS:
    if os.path.exists(f"stiefel_seq_alpha_{a:.2f}_n{n}_k{k_eff}_M{M_local}.pkl"):
        valid_alphas.append(a)

if not valid_alphas:
    print(f"Error: No .pkl files found for M_local={M_local}. Check your folder!")
    exit()

print(f"Found {len(valid_alphas)} valid data files. Pre-generating all ensembles...")

# Global Storage Dictionaries
all_samples = {}
all_splines = {}

# Pre-generate everything BEFORE opening the UI
for alpha in valid_alphas:
    # 1. Generate the Monte Carlo samples
    all_samples[alpha] = generate_ensemble(alpha, NUM_SAMPLES)

    # 2. Extract the Theoretical PDF Spline for Theta 1
    filename = f"stiefel_seq_alpha_{alpha:.2f}_n{n}_k{k_eff}_M{M_local}.pkl"
    with open(filename, 'rb') as f:
        root_node = pickle.load(f)['sparse_tree']

    t_nodes_true = np.concatenate((root_node['vals'], [0.0]))[::-1]
    u_nodes_true = np.concatenate(([0.0], root_node['cdf']))[::-1]

    unique_idx_true = np.unique(t_nodes_true, return_index=True)[1]
    t_nodes_true = t_nodes_true[unique_idx_true]
    u_nodes_true = u_nodes_true[unique_idx_true]

    cdf_spline = PchipInterpolator(t_nodes_true, u_nodes_true)
    all_splines[alpha] = cdf_spline.derivative()

print("Pre-generation complete. Launching interactive UI...")

# =============================================================================
# 4. INTERACTIVE MULTI-ALPHA VISUALIZATION
# =============================================================================
# Global UI State
state = {
    'alpha': valid_alphas[0],
    'angle_idx': 0
}

# --- UI Setup ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(left=0.25, bottom=0.15)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, k_eff))

def draw_histogram():
    ax.clear()

    # Instantly grab data from pre-generated dictionaries
    alpha = state['alpha']
    angle_idx = state['angle_idx']

    data = all_samples[alpha][:, angle_idx]

    # Adaptive Bounding
    p_low = np.percentile(data, 0.5)
    p_high = np.percentile(data, 99.5)
    margin = max((p_high - p_low) * 0.1, 0.02)

    plot_min = max(0.0, p_low - margin)
    plot_max = min(np.pi/2, p_high + margin)

    custom_bins = np.linspace(plot_min, plot_max, BINS)

    # Plot Histogram
    ax.hist(data, bins=custom_bins, density=True,
            color=colors[angle_idx], alpha=0.9, edgecolor='white', linewidth=0.5,
            label=f'Monte Carlo Samples (M={M_local})')

    # Draw Theoretical Spline Overlay (Theta 1 Only)
    if angle_idx == 0:
        x_smooth = np.linspace(plot_min, plot_max, 500)
        y_raw = np.abs(all_splines[alpha](x_smooth))
        y_smooth = gaussian_filter1d(y_raw, sigma=15)

        ax.plot(x_smooth, y_smooth, color='crimson', linewidth=3,
                label='Theoretical PDF (Smoothed Gibbs Measure)')
        ax.legend(loc='upper left', fontsize=11)

    ax.set_title(f"Marginal Density of $\\theta_{{{angle_idx+1}}}$ ($\\alpha={alpha}$)", fontsize=15, fontweight='bold')
    ax.set_xlim(plot_min, plot_max)
    ax.set_xlabel('Angle (Radians)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    fig.canvas.draw_idle()

# --- Interactive Menus ---

# 1. Alpha Menu (Top Left)
ax_menu_alpha = plt.axes([0.05, 0.65, 0.12, 0.25])
ax_menu_alpha.set_facecolor('whitesmoke')
ax_menu_alpha.set_title("Select Alpha", fontweight='bold')
alpha_labels = [str(a) for a in valid_alphas]
radio_alpha = RadioButtons(ax_menu_alpha, alpha_labels, active=0)

def on_alpha_click(label):
    state['alpha'] = float(label)
    draw_histogram()

radio_alpha.on_clicked(on_alpha_click)

# 2. Angle Menu (Bottom Left)
ax_menu_angle = plt.axes([0.05, 0.15, 0.12, 0.45])
ax_menu_angle.set_facecolor('whitesmoke')
ax_menu_angle.set_title("Select Angle", fontweight='bold')
angle_labels = [f"Theta {i+1}" for i in range(k_eff)]
radio_angle = RadioButtons(ax_menu_angle, angle_labels, active=0)

def on_angle_click(label):
    state['angle_idx'] = angle_labels.index(label)
    draw_histogram()

radio_angle.on_clicked(on_angle_click)

# Draw initial state and show
draw_histogram()
plt.show()