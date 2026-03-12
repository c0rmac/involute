import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# CONFIGURATION PARAMETERS
# Modify these variables to select your files
# Set to None to act as a wildcard (*)
# ==========================================
#DIRECTORY = "../../../cmake-build-release-mlx/tests/ackley_results"          # e.g., "data_folder" or "." for current directory
DIRECTORY = "../../../cmake-build-release-mlx/tests/schwefel_results"          # e.g., "data_folder" or "." for current directory
DIMENSION = 3            # e.g., 3, or None for all dimensions
TYPE = "Custom"      # e.g., "Aggressive", or None for all types
      # e.g., "Aggressive", or None for all types
RUN = 13                  # e.g., 0, or None for all runs
# ==========================================

def plot_solver_data(filepath):
    """Reads a CSV and plots energy, beta (log), and lambda (log) against step."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    required_cols = ['step', 'energy', 'beta', 'lambda']
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping {filepath}: Missing one or more required columns ({required_cols}).")
        return

    filename = os.path.basename(filepath)

    # Create 3 separate subplots sharing the same X-axis
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Solver Metrics: {filename}", fontsize=16)

    # 1. Plot Energy (Linear Scale)
    axes[0].plot(df['step'], df['energy'], color='tab:blue', linewidth=2)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title('Energy vs Step', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Plot Beta (Log Scale)
    axes[1].plot(df['step'], df['beta'], color='tab:orange', linewidth=2)
    axes[1].set_yscale('log')  # Set y-axis to logarithmic scale
    axes[1].set_ylabel('Beta (log scale)', fontsize=12)
    axes[1].set_title('Beta vs Step', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Plot Lambda (Log Scale)
    axes[2].plot(df['step'], df['lambda'], color='tab:green', linewidth=2)
    axes[2].set_yscale('log')  # Set y-axis to logarithmic scale
    axes[2].set_ylabel('Lambda (log scale)', fontsize=12)
    axes[2].set_xlabel('Step', fontsize=12)
    axes[2].set_title('Lambda vs Step', fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def main():
    # Construct the file pattern based on hardcoded variables
    d_val = DIMENSION if DIMENSION is not None else '*'
    type_val = TYPE if TYPE is not None else '*'
    run_val = RUN if RUN is not None else '*'

    # Expected filename: d=3_type=Aggressive_run=0.solver.csv
    file_pattern = f"d={d_val}_type={type_val}_run={run_val}.solver.csv"
    search_pattern = os.path.join(DIRECTORY, file_pattern)

    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No files matching pattern '{file_pattern}' found in directory: {DIRECTORY}")
        return

    print(f"Found {len(csv_files)} matching file(s). Generating plots...")

    for file in csv_files:
        plot_solver_data(file)

if __name__ == "__main__":
    main()