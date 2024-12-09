"""
Courtesy of Alex Amice.
"""
from pydrake.all import (
    VPolytope,
    HPolyhedron,
)

import numpy as np
import scipy.spatial as spatial
from scipy.stats.qmc import LatinHypercube

from utils import *


def generate_test_2D(filename, low_bound, high_bound, resolution, num_sets):
    """
    Fairly overcomplicated function to generate test examples for 2D GCS traj 
    opt problems.
    
    The resulting test is visualized and written to `filename`.
    """
    def write_test_to_file(filename, As, bs, s, t, N, M):
        with open(filename, 'w') as f:
            # Write imports
            f.write("import numpy as np\n")
            f.write("import os\n")
            f.write("import sys\n\n")

            f.write("path_to_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n")
            f.write("sys.path.append(path_to_utils)\n")
            f.write("from utils import convert_pt_to_polytope, visualize_results\n\n")
            
            # Write source and target points
            f.write(f"s = np.array({s.tolist()})\n\n")
            f.write(f"t = np.array({t.tolist()})\n\n")
            
            # Write conversion of points to polytopes
            f.write("A_s, b_s = convert_pt_to_polytope(s, eps=1e-6)\n")
            f.write("A_t, b_t = convert_pt_to_polytope(t, eps=1e-6)\n\n")
            
            # Write A and b matrices for numerical keys
            numerical_keys = sorted([k for k in As.keys() if isinstance(k, int)])
            for key in numerical_keys:
                A = As[key]
                b = bs[key]
                f.write(f"A{key} = np.array({A.tolist()})\n\n")
                f.write(f"b{key} = np.array({b.tolist()})\n\n")
            
            # Assemble As dictionary
            f.write("As = {\n")
            f.write("    \"s\": A_s,\n")
            f.write("    \"t\": A_t,\n")
            for key in numerical_keys:
                f.write(f"    {key}: A{key},\n")
            f.write("}\n\n")
            
            # Assemble bs dictionary
            f.write("bs = {\n")
            f.write("    \"s\": b_s,\n")
            f.write("    \"t\": b_t,\n")
            for key in numerical_keys:
                f.write(f"    {key}: b{key},\n")
            f.write("}\n\n")
            
            # Define variable n
            if numerical_keys:
                first_key = numerical_keys[0]
                f.write(f"n = A{first_key}.shape[1]\n")
            else:
                f.write("n = 0  # No polytopes defined\n\n\n")
                
            f.write("# For rounding step: \n")
            f.write(f"N = {N}\n")
            f.write(f"M = {M}\n\n")
                
            f.write("# If file is run directly, visualize the GCS\n")
            f.write("if __name__ == \"__main__\":\n")
            f.write("   visualize_results(As, bs, {**{i: 0 for i in range(len(As)-2)}, **{\"s\": np.hstack([s,s]), \"t\": np.hstack([t,t])}}, {**{i: 0 for i in range(len(As)-2)}, **{\"s\": 1, \"t\": 1}})")
        
    def generate_random_point_in_hpoly(hpolyhedron):
        """
        Generate a random point inside an HPolyhedron.
        """
        A = hpolyhedron.A()
        b = hpolyhedron.b()
        dim = A.shape[1]

        # Sample a random point within the bounds defined by A * x <= b
        # For simplicity, using rejection sampling
        while True:
            candidate_point = np.random.uniform(low_bound, high_bound, size=dim)  # Adjust range as needed
            if np.all(A @ candidate_point <= b):
                return candidate_point
            
    grid_x_size = int((high_bound - low_bound) / resolution)
    grid_y_size = int((high_bound - low_bound) / resolution)
    x = np.linspace(low_bound, high_bound, grid_x_size)
    y = np.linspace(low_bound, high_bound, grid_y_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack((X.flatten(), Y.flatten())).T
    
    # Randomly select seed points for regions    
    seed_indices = np.random.choice(len(grid_points), num_sets, replace=False)
    seeds = grid_points[seed_indices]
    lhs = LatinHypercube(d=2, optimization="lloyd")
    seeds = lhs.random(n=num_sets)
    seeds[:,0] = (high_bound - low_bound) * seeds[:,0] + low_bound
    seeds[:,1] = (high_bound - low_bound) * seeds[:,1] + low_bound
    distances = spatial.distance.cdist(seeds, seeds)
    distances[distances == 0] = np.inf

    seed_radius = np.min(distances, axis=1)
    hpolyhedrons = []
    As = dict()
    bs = dict()
    for i, (seed, radius) in enumerate(zip(seeds, seed_radius)):
        # Compute distances to the given point
        local_distances = np.linalg.norm(grid_points - seed, axis=1)

        # Find points within the threshold
        close_points = grid_points[local_distances <= radius]
        potential_vertices = close_points[
            np.random.choice(
                len(close_points), size=int(0.3 * len(close_points)), replace=False
            )
        ]
        vpoly = None
        while close_points.shape[0] < 3 or vpoly is None:
            radius *= 1.1
            close_points = grid_points[local_distances <= radius]
            potential_vertices = close_points[
                np.random.choice(
                    len(close_points), size=int(0.1 * len(close_points)), replace=False
                )
            ]
            if potential_vertices.shape[0] > 3:
                try:
                    vpoly_tmp = VPolytope(
                        potential_vertices.T
                    ).GetMinimalRepresentation()
                    if vpoly_tmp.CalcVolume() > 1e-5:
                        vpoly = vpoly_tmp
                except ValueError:
                    vpoly = None
        new_hpoly = HPolyhedron(VPolytope(potential_vertices.T))
        hpolyhedrons.append(new_hpoly)  # not efficient but it works
        As[i] = new_hpoly.A()
        bs[i] = new_hpoly.b()
    
    # Randomly select two different hpolyhedrons for source and target
    hpoly_indices = np.random.choice(len(hpolyhedrons), size=2, replace=False)
    source_hpoly = hpolyhedrons[hpoly_indices[0]]
    target_hpoly = hpolyhedrons[hpoly_indices[1]]

    # Generate random source and target points within the selected hpolyhedrons
    x_s = generate_random_point_in_hpoly(source_hpoly)
    x_t = generate_random_point_in_hpoly(target_hpoly)
    
    A_s, b_s = convert_pt_to_polytope(x_s, eps=1e-6)
    A_t, b_t = convert_pt_to_polytope(x_t, eps=1e-6)
    
    x_s = np.hstack((x_s, x_s))
    x_t = np.hstack((x_t, x_t))
    
    As = {**As, **{"s": A_s, "t": A_t}}
    bs = {**bs, **{"s": b_s, "t": b_t}}
    
    visualize_results(As, bs, {**{i: 0 for i in range(seeds.shape[0])}, **{"s": x_s, "t": x_t}}, {**{i: 0 for i in range(seeds.shape[0])}, **{"s": 1, "t": 1}})
    
    write_test_to_file(filename, As, bs, x_s[:2], x_t[:2], int(num_sets/5), int(2*num_sets/5))

    
generate_test_2D("test_data/benchmark4.py", -60, 60, 4, 40)