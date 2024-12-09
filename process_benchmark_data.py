import os
import pickle

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
        
        
def plot_sequences(benchmark_num, data_files, sequence_key, ylabel, title_suffix, save_dir="benchmark_data"):
    """
    Plots a specific sequence (e.g., rho_seq) for all data files in a benchmark.

    Args:
        benchmark_num (str): The benchmark number as a string.
        data_files (list): List of data entries for the benchmark.
        sequence_key (str): The key in the data dictionary to plot.
        ylabel (str): Label for the Y-axis.
        title_suffix (str): Suffix for the plot title.
        save_dir (str): Directory to save the plots.
    """
    plt.figure(figsize=(10, 6))
    
    for entry in data_files:
        data_file_name = entry['file_name']
        data = entry['data']
        
        # Check if ADMM data exists
        if not data.get('ADMM', False):
            continue
        
        # Check if the sequence exists
        sequence = data.get(sequence_key)
        if sequence is None:
            print(f"Warning: '{sequence_key}' not found in {data_file_name}. Skipping.")
            continue
        
        # Plot the sequence
        plt.plot(sequence, label=data_file_name)
    
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(f'Benchmark {benchmark_num} - {title_suffix}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = f"benchmark{benchmark_num}_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close()
    print(f"Saved plot: {os.path.join(save_dir, plot_filename)}")    
    


# Iterate through each benchmark and generate the plots
for data_file_name, data_dict in loaded_data.items():
    # print(data_file_name)
    print(data_dict)
    # Visualize results for each benchmark
    visualize_results(data['As'], data['bs'], data['x_v_sol'], data['y_v_sol'], data['x_v_rounded'], data['y_v_rounded'], save_to_file=f"benchmark_data/{data_file_name}.png")
    
    benchmark_num = data_file_name[-1]
    
    # Plot rho_seq
    plot_sequences(
        benchmark_num=benchmark_num,
        data_files=data_files,
        sequence_key='rho_seq',
        ylabel='Rho Sequence',
        title_suffix='Rho Sequence'
    )
    
    # Plot pri_res_seq
    plot_sequences(
        benchmark_num=benchmark_num,
        data_files=data_files,
        sequence_key='pri_res_seq',
        ylabel='Primal Residual Sequence',
        title_suffix='Primal Residual Sequence'
    )
    
    # Plot dual_res_seq
    plot_sequences(
        benchmark_num=benchmark_num,
        data_files=data_files,
        sequence_key='dual_res_seq',
        ylabel='Dual Residual Sequence',
        title_suffix='Dual Residual Sequence'
    )