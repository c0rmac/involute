import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
IMAGE_NAME = "./schwefel_viz/schwefel"
DIRECTORY = "schwefel_results_adam"             # Directory containing the CSV files
DIMENSION = 5               # Specify the dimension you want to plot
TYPE = "Custom"         # Specify the type, or None for all types

# Exclusion criteria
EXCLUSION_METRIC = "energy" # Metric to check for exclusion (e.g., 'energy', 'beta', or 'lambda')
MAX_FINAL_VALUE = 0.05     # Exclude runs if their LAST value in EXCLUSION_METRIC is strictly above this
# ==========================================

def main():
    # Construct the search pattern
    d_val = DIMENSION if DIMENSION is not None else '*'
    type_val = TYPE if TYPE is not None else '*'

    file_pattern = f"d={d_val}_type={type_val}_run=*.solver.csv"
    search_pattern = os.path.join(DIRECTORY, file_pattern)
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No files matching pattern '{file_pattern}' found in directory: {DIRECTORY}")
        return

    print(f"Found {len(csv_files)} matching file(s). Processing data...")

    valid_dfs = []
    excluded_count = 0

    # 1. Read files and apply exclusion filter
    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # Check the final value of the specified metric
            final_value = df[EXCLUSION_METRIC].iloc[-1]
            if final_value > MAX_FINAL_VALUE:
                print(f"  -> Excluding {os.path.basename(file)} (Final {EXCLUSION_METRIC} = {final_value:.4f})")
                excluded_count += 1
            else:
                valid_dfs.append(df)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"Successfully loaded {len(valid_dfs)} runs. Excluded {excluded_count} runs.")

    if not valid_dfs:
        print("No valid runs left to plot after applying the exclusion threshold.")
        return

    # 2. Combine all valid DataFrames and calculate aggregations
    # Concatenate all dataframes, then group by 'step' to handle any missing/misaligned steps
    all_data = pd.concat(valid_dfs)

    # Calculate mean, min, and max for each step
    agg_df = all_data.groupby('step').agg(['mean', 'min', 'max'])

    # 3. Create the plots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    #fig.suptitle(f"Aggregated Runs (d={DIMENSION}, type={TYPE})\n{len(valid_dfs)} Runs Averaged", fontsize=16)
    fig.suptitle(f"SO({DIMENSION}) - Run Type: {TYPE}", fontsize=16)

    metrics = [
        ('energy', 'tab:blue', 'Energy', axes[0], False),
        ('beta', 'tab:orange', 'Beta (log scale)', axes[1], True),
        ('lambda', 'tab:green', 'Lambda (log scale)', axes[2], True)
    ]

    for metric_name, line_color, ylabel, ax, is_log in metrics:
        steps = agg_df.index
        mean_vals = agg_df[metric_name]['mean']
        min_vals = agg_df[metric_name]['min']
        max_vals = agg_df[metric_name]['max']

        # Plot the shaded min/max area (using a light blue shade per your request)
        ax.fill_between(steps, min_vals, max_vals, color='tab:blue', alpha=0.2, label='Min/Max Range')

        # Plot the average line
        ax.plot(steps, mean_vals, color=line_color, linewidth=2, label='Average')

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{metric_name.capitalize()} vs Step', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        if is_log:
            ax.set_yscale('log')

    axes[2].set_xlabel('Step', fontsize=12)

    plt.tight_layout()
    #plt.show()
    plt.savefig(IMAGE_NAME + "_d=" + str(DIMENSION) + "_type=" + TYPE + ".jpg")

if __name__ == "__main__":
    main()