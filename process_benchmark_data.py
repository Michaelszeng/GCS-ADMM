import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from utils import *

# Load and store data from all benchmarks
benchmark_data_dir = "benchmark_data"
loaded_data = {}
for filename in os.listdir(benchmark_data_dir):
    if filename.endswith('.pkl'):
        file_path = os.path.join(benchmark_data_dir, filename)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a variable name based on the filename (without extension)
        variable_name = os.path.splitext(filename)[0]
        
        loaded_data[variable_name] = data


################################################################################
##### Generate 2D Visuals
################################################################################
for data_file_name, data in loaded_data.items():    
    # Visualize results for each benchmark
    visualize_results(data['As'], data['bs'], data['x_v_sol'], data['y_v_sol'], data['x_v_rounded'], data['y_v_rounded'], save_to_file=f"benchmark_data/plots/{data_file_name}.png")
    
    
################################################################################
##### Plot Primal & Dual Residuals
################################################################################
# Predefined order and colors for algorithms
algorithm_order = ["v1", "v2", "v3"]  # Adjust based on your naming conventions
algorithm_labels = {
    "v1": "Vertex-Edge Split 1",
    "v2": "Vertex-Edge Split 2",
    "v3": "Full Vertex Split"
}
colormap = matplotlib.colormaps['tab10']
algorithm_colors = {alg: colormap(3 * i) for i, alg in enumerate(algorithm_order)}  # 3*i to get distinct colors

# Group data by benchmarks
benchmarks = {}
for data_file_name, data in loaded_data.items():
    # Extract benchmark identifier (assuming format "<algorithm_name>_benchmark<i>")
    benchmark_name = data_file_name.rsplit('_', 1)[-1]
    if benchmark_name not in benchmarks:
        benchmarks[benchmark_name] = {}
    benchmarks[benchmark_name][data_file_name] = data    

# Generate Primal Residual & Dual Residual Plots
for benchmark_name, algorithms_data in benchmarks.items():
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axs[0].set_title(f"Benchmark {benchmark_name}: Primal Residuals", fontsize=14)
    axs[1].set_title(f"Benchmark {benchmark_name}: Dual Residuals", fontsize=14)
    
    for idx, algorithm_key in enumerate(algorithm_order):
        # Check if the algorithm exists in this benchmark
        matched_data = None
        for algorithm_name, data in algorithms_data.items():
            if algorithm_key in algorithm_name:
                matched_data = data
                break
        
        if matched_data and matched_data.get("ADMM"):  # Only plot if ADMM data is present
            pri_res_seq = matched_data.get("pri_res_seq", [])[1:]  # Skip the first residual value
            dual_res_seq = matched_data.get("dual_res_seq", [])[1:]  # Skip the first residual value
            iterations = matched_data.get("iterations", len(pri_res_seq))  # Use provided iterations or sequence length
            
            label = algorithm_labels.get(algorithm_key, algorithm_key)
            color = algorithm_colors[algorithm_key]
            
              # just offset blue line to make it look better
            if algorithm_key == "v1":
                offset = -0.5
            else:
                offset = 0
            
            # Plot primal residuals
            line_pri, = axs[0].plot(pri_res_seq, label=label, color=color)
            if len(pri_res_seq) > 0:
                x_end_pri = len(pri_res_seq) - 1
                y_end_pri = pri_res_seq[-1]
                axs[0].text(x_end_pri, y_end_pri * (1 + offset), f"{iterations} iters", color=color, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left')
            
            # Plot dual residuals
            line_dual, = axs[1].plot(dual_res_seq, label=label, color=color)
            if len(dual_res_seq) > 0:
                x_end_dual = len(dual_res_seq) - 1
                y_end_dual = dual_res_seq[-1]
                axs[1].text(x_end_dual, y_end_dual * (1 + offset), f"{iterations} iters", color=color, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left')

    # Set log scale for both axes
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    
    # Add legends, labels, and grid
    for ax in axs:
        ax.legend(fontsize=10)
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("Residual Value", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"benchmark_data/plots/{benchmark_name}_residuals.png")
    plt.close()
    
    
################################################################################
##### Plot Solve Times
################################################################################
# Predefined order and colors for algorithms
algorithm_order = ["classic", "v1", "v2", "v3"]  # Adjust based on your naming conventions
algorithm_labels = {
    "classic": "Commercial Solver",
    "v1": "Vertex-Edge Split 1",
    "v2": "Vertex-Edge Split 2",
    "v3": "Full Vertex Split"
}
colormap = matplotlib.colormaps['tab10']
algorithm_colors = {alg: colormap(3*(i-1)) for i, alg in enumerate(algorithm_order)}  # 3*i to get some nice blue, red, pink, (i-1) so that classic gets a different color
algorithm_colors["classic"] = colormap(1)  # Set classic solver color]

# Prepare data for solve time plot
benchmark_ids = sorted(benchmarks.keys(), key=lambda x: int(x[-1]))  # Sort benchmarks numerically by ID
solve_times = {alg: [] for alg in algorithm_order}  # Initialize solve time data for each algorithm

# Collect solve times for each algorithm across benchmarks
for benchmark_id in benchmark_ids:
    algorithms_data = benchmarks[benchmark_id]
    for algorithm_key in algorithm_order:
        # Match algorithm key to data
        matched_data = None
        for algorithm_name, data in algorithms_data.items():
            if algorithm_key in algorithm_name:  # Match algorithm_key to its corresponding data
                matched_data = data
                break
        
        if matched_data:
            solve_times[algorithm_key].append(matched_data.get("solve_time", None))
        else:
            solve_times[algorithm_key].append(None)  # Fill with None if no data for the algorithm

# Define custom labels for x-axis
custom_labels = ["4 Vertices", "8 Vertices", "20 Vertices", "40 Vertices"]

# Plot solve times
plt.figure(figsize=(8, 4))
x_positions = range(1, len(benchmark_ids) + 1)  # Benchmarks as x-axis positions

for algorithm_key in algorithm_order:
    y_values = solve_times[algorithm_key]
    label = algorithm_labels.get(algorithm_key, algorithm_key)
    color = algorithm_colors[algorithm_key]
    
    # Mask None values to avoid plotting gaps
    mask = [v is not None for v in y_values]
    x_vals = np.array(x_positions)[mask]
    y_vals = np.array(y_values)[mask]
    
    plt.plot(x_vals, y_vals, label=label, color=color, marker='o', linestyle='-')

# Customize plot
plt.title("Solve Times versus Problem Size", fontsize=16)
plt.xlabel("Problem Size (# Vertices in GCS)", fontsize=14)
plt.ylabel("Solve Time (s)", fontsize=14)
plt.yscale("log")
plt.xticks(ticks=x_positions, labels=custom_labels, fontsize=12)  # Use custom labels
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Save and show plot
plt.tight_layout()
plt.savefig("benchmark_data/plots/solve_times_across_benchmarks.png")
plt.close()


################################################################################
##### Table for Cost Comparison
################################################################################
# Prepare data for the table
benchmark_ids = sorted(benchmarks.keys(), key=lambda x: int(x[-1]))  # Sort benchmarks numerically by ID
table_data = {alg: [] for alg in algorithm_order}  # Initialize data for each algorithm

# Collect costs for each algorithm across benchmarks
for benchmark_id in benchmark_ids:
    algorithms_data = benchmarks[benchmark_id]
    for algorithm_key in algorithm_order:
        # Match algorithm key to data
        matched_data = None
        for algorithm_name, data in algorithms_data.items():
            if algorithm_key in algorithm_name:
                matched_data = data
                break
        
        if matched_data:
            table_data[algorithm_key].append(matched_data.get("cost", None))  # Append cost
        else:
            table_data[algorithm_key].append(None)  # Append None if no data for the algorithm

# Create a DataFrame for the table
table_df = pd.DataFrame(table_data, index=benchmark_ids)

# Rename columns to use algorithm labels
table_df.rename(columns=algorithm_labels, inplace=True)

# Rename the index to be more descriptive
table_df.index.name = "Benchmark"

# Truncate numbers to 3 decimal places
table_df = table_df.round(3)

# Plot the table as an image
fig, ax = plt.subplots(figsize=(10, len(table_df) * 0.6))  # Adjust figure size based on rows
ax.axis('tight')
ax.axis('off')

# Render the table
table = ax.table(cellText=table_df.values,
                 colLabels=table_df.columns,
                 rowLabels=table_df.index,
                 loc='center',
                 cellLoc='center',
                 colLoc='center')

# Apply styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)  # Scale up the table for better readability

# Add custom styling to header cells
for (row, col), cell in table.get_celld().items():
    if row == 0 or col == -1:  # Header cells
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#630101')  # Green for headers
    else:
        cell.set_facecolor('#F8F8F8')  # Light gray for other cells
    cell.set_edgecolor('black')  # Add borders

# Save the table as an image
plt.savefig("benchmark_data/plots/cost_table.png", dpi=300, bbox_inches='tight')
plt.close()