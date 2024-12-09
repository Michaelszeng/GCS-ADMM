import os
import pickle
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
# Group data by benchmarks
benchmarks = {}
solve_times = {}
for data_file_name, data in loaded_data.items():
    # Extract benchmark identifier (assuming format "<algorithm_name>_benchmark<i>")
    benchmark_name = data_file_name.rsplit('_', 1)[-1]
    if benchmark_name not in benchmarks:
        benchmarks[benchmark_name] = {}
    benchmarks[benchmark_name][data_file_name] = data

# Generate Primal Residual & Dual Residual Plots
colormap = matplotlib.colormaps['tab10']  # Updated API for colormap access
for benchmark_name, algorithms_data in benchmarks.items():
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axs[0].set_title(f"Benchmark {benchmark_name}: Primal Residuals", fontsize=14)
    axs[1].set_title(f"Benchmark {benchmark_name}: Dual Residuals", fontsize=14)
    
    # Cycle through colors for each algorithm
    color_idx = 3  # start with red
    
    for algorithm_name, data in algorithms_data.items():
        if data.get("ADMM"):  # Only plot if ADMM data is present
            pri_res_seq = data.get("pri_res_seq", [])[1:]  # Skip the first residual value (initialization)
            dual_res_seq = data.get("dual_res_seq", [])[1:]  # Skip the first residual value (initialization)
            
            # Assign a color from the colormap
            color = colormap(color_idx % len(colormap.colors))
            color_idx += 1
            
            # Plot primal residuals on a log scale
            if "v1" in algorithm_name:
                algorithm_name = "vertex-edge split 1"
            elif "v2" in algorithm_name:
                algorithm_name = "vertex-edge split 2"
            elif "v3" in algorithm_name:
                algorithm_name = "full vertex split"
            
            axs[0].plot(pri_res_seq, label=f"{algorithm_name}", color=color)
            
            # Plot dual residuals on a log scale
            axs[1].plot(dual_res_seq, label=f"{algorithm_name}", color=color)

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
# Collect solve times for each algorithm across benchmarks
solve_times = {}
benchmarks_list = sorted(benchmarks.keys(), key=lambda x: int(x.strip('benchmark')))
benchmarks_list = ["benchmark1"]  # TEMPORARY

for benchmark_name, algorithms_data in benchmarks.items():
    for algorithm_name, data in algorithms_data.items():
        if algorithm_name not in solve_times:
            solve_times[algorithm_name] = []
        solve_times[algorithm_name].append(data.get("solve_time", None))

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

colormap = plt.cm.tab10
colors = [colormap(i) for i in range(len(solve_times))]

# Plot solve times for each algorithm
for i, (algorithm_name, times) in enumerate(solve_times.items()):
    # Ensure times are in the correct order of benchmarks
    times = [times[benchmarks_list.index(f"benchmark{j+1}")] for j in range(len(benchmarks_list))]
    ax.plot(benchmarks_list, times, marker='o', label=algorithm_name, color=colors[i])

# Add labels, legend, and grid
ax.set_title("Solve Times Across Benchmarks", fontsize=16)
ax.set_xlabel("Benchmarks", fontsize=14)
ax.set_ylabel("Solve Time (s)", fontsize=14)
ax.set_xticks(benchmarks_list)
ax.set_xticklabels([f"Benchmark {i+1}" for i in range(len(benchmarks_list))])
ax.legend(fontsize=12, title="Algorithms")
ax.grid(True, linestyle='--', alpha=0.6)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("benchmark_data/plots/solve_times_across_benchmarks.png")
plt.close()