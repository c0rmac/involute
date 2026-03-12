import os
import glob
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# ==========================================
# CONFIGURATION & FORMULAS
# ==========================================
OUTPUT_FILE = "benchmark.md"
SUCCESS_THRESHOLD = 0.05
RESULTS_SUFFIX = "_results_adam"
VIZ_SUFFIX = "_viz"
IMG_DIR = "./analysis/results/"

# Extensible dictionary for Markdown function definitions (LaTeX formulas).
FORMULAS = {
    "ackley": r"""
$$ f(X) = -a \exp\left(-b \sqrt{\frac{1}{n} \sum (X - I)^2}\right) - \exp\left(\frac{1}{n} \sum \cos(c (X - I))\right) + a + \exp(1) $$

Where $a=20$, $b=0.2$, $c=2\pi$, and $n=d^2$.
""",
    "schwefel": r"""
$$ f(X) = 418.9829n - \sum Z \sin(\sqrt{|Z|}) $$

Where $Z = 250(X - I) + 420.968746$, and $n=d^2$.
"""
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_run_stats(results_dir, d_val, type_val):
    """Parses CSVs for a specific dimension and type to calculate success metrics."""
    csv_pattern = f"d={d_val}_type={type_val}_run=*.solver.csv"
    search_pattern = os.path.join(results_dir, csv_pattern)
    csv_files = glob.glob(search_pattern)

    total_runs = len(csv_files)
    if total_runs == 0:
        return {"total": 0, "success_rate": 0.0, "mean_steps": 0.0, "median_steps": 0.0}

    successful_runs = 0
    steps_taken = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            final_energy = df['energy'].iloc[-1]

            if final_energy < SUCCESS_THRESHOLD:
                successful_runs += 1
                if 'step' in df.columns:
                    steps_taken.append(df['step'].iloc[-1])
                else:
                    steps_taken.append(len(df))
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
    mean_steps = np.mean(steps_taken) if steps_taken else 0
    median_steps = np.median(steps_taken) if steps_taken else 0

    return {
        "total": total_runs,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "median_steps": median_steps
    }

def format_anchor(text):
    """Converts a string into a standard markdown anchor link."""
    return text.lower().replace(" ", "-").replace("=", "").replace(":", "")

# ==========================================
# MAIN SCRIPT
# ==========================================
def process_benchmarks():
    # Data structure: dict[func_name][dimension][run_type] = { image_path, stats }
    benchmark_data = defaultdict(lambda: defaultdict(dict))

    viz_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.endswith(VIZ_SUFFIX)]
    if not viz_dirs:
        print(f"No directories ending with '{VIZ_SUFFIX}' found.")
        return

    print("Scanning directories and parsing results...")

    # 1. Build the data structure
    for viz_dir in sorted(viz_dirs):
        func_name = viz_dir.replace(VIZ_SUFFIX, "")
        results_dir = f"{func_name}{RESULTS_SUFFIX}"

        if not os.path.exists(results_dir):
            print(f"Warning: Missing results directory {results_dir} for {func_name}")
            continue

        images = glob.glob(os.path.join(viz_dir, "*.jpg"))
        for image_path in images:
            filename = os.path.basename(image_path)
            match = re.search(r"d=(\d+)_type=([^_.]+)", filename)

            if match:
                d_val = match.group(1)
                type_val = match.group(2)
                stats = get_run_stats(results_dir, d_val, type_val)
                benchmark_data[func_name][d_val][type_val] = {
                    "image": IMG_DIR + image_path,
                    "stats": stats
                }

    if not benchmark_data:
        print("No valid benchmark data found.")
        return

    # 2. Generate Markdown
    markdown_lines = ["# SO(d) Solver Benchmarks\n\n"]

    # --- TABLE OF CONTENTS ---
    markdown_lines.append("## Table of Contents\n")
    for func_name in sorted(benchmark_data.keys()):
        func_title = f"{func_name.capitalize()} Function"
        markdown_lines.append(f"- [{func_title}](#{format_anchor(func_title)})\n")

        # Sort dimensions numerically for the ToC
        sorted_dims = sorted(benchmark_data[func_name].keys(), key=lambda x: int(x))
        for d_val in sorted_dims:
            dim_title = f"{func_name.capitalize()} - Dimension {d_val}"
            markdown_lines.append(f"  - [Dimension {d_val}](#{format_anchor(dim_title)})\n")
    markdown_lines.append("\n---\n\n")

    # --- BODY CONTENT ---
    for func_name in sorted(benchmark_data.keys()):
        func_title = f"{func_name.capitalize()} Function"
        markdown_lines.append(f"## {func_title}\n\n")

        if func_name in FORMULAS:
            markdown_lines.append(FORMULAS[func_name])
            markdown_lines.append("\n")

        sorted_dims = sorted(benchmark_data[func_name].keys(), key=lambda x: int(x))
        for d_val in sorted_dims:
            dim_title = f"{func_name.capitalize()} - Dimension {d_val}"
            markdown_lines.append(f"### {dim_title}\n\n")

            types_data = benchmark_data[func_name][d_val]
            sorted_types = sorted(types_data.keys())

            # Build Markdown Table Headers
            headers = " | ".join([f"Type: {t}" for t in sorted_types])
            separators = " | ".join(["---"] * len(sorted_types))
            markdown_lines.append(f"| {headers} |\n")
            markdown_lines.append(f"| {separators} |\n")

            # Build Markdown Table Image Row
            images_row = " | ".join([f"![{func_name} d={d_val} type={t}]({types_data[t]['image']})" for t in sorted_types])
            markdown_lines.append(f"| {images_row} |\n")

            # Build Markdown Table Stats Row
            stats_cells = []
            for t in sorted_types:
                st = types_data[t]["stats"]
                stats_str = (
                    f"**Runs:** {st['total']}<br>"
                    f"**Success:** {st['success_rate']:.1f}%<br>"
                    f"**Mean Steps:** {st['mean_steps']:.1f}<br>"
                    f"**Median Steps:** {st['median_steps']:.1f}"
                )
                stats_cells.append(stats_str)

            stats_row = " | ".join(stats_cells)
            markdown_lines.append(f"| {stats_row} |\n\n")
            markdown_lines.append("---\n\n")

    # 3. Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(markdown_lines)

    print(f"Successfully generated {OUTPUT_FILE}!")

if __name__ == "__main__":
    process_benchmarks()