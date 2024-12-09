import os
import pickle

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

for var_name, data in loaded_data.items():
    print(f"Data from {var_name}:")
    print(data)
    print()
    
    # Plot 1: Visualize results for each benchmark
    
    
