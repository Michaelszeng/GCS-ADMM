"""
Courtesy of Alex Amice.
"""
from pydrake.all import (
    VPolytope,
)

import numpy as np
import scipy.spatial as spatial
from scipy.stats.qmc import LatinHypercube


def generate_grid_path_problem(x_low, x_high, y_low, y_high, resolution, num_sets):
    grid_x_size = int((x_high - x_low) / resolution)
    grid_y_size = int((y_high - y_low) / resolution)
    x = np.linspace(x_low, x_high, grid_x_size)
    y = np.linspace(y_low, y_high, grid_y_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack((X.flatten(), Y.flatten())).T

    # Randomly select seed points for regions
    seed_indices = np.random.choice(len(grid_points), num_sets, replace=False)
    seeds = grid_points[seed_indices]
    lhs = LatinHypercube(d=2, optimization="lloyd")
    seeds = lhs.random(n=num_sets)
    seeds[:,0] = (x_high - x_low) * seeds[:,0] + x_low
    seeds[:,1] = (y_high - y_low) * seeds[:,1] + y_low
    #
    distances = spatial.distance.cdist(seeds, seeds)
    distances[distances == 0] = np.inf

    seed_radius = np.min(distances, axis=1)
    vpolytopes = []
    for seed, radius in zip(seeds, seed_radius):
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
        vpolytopes.append(VPolytope(potential_vertices.T))