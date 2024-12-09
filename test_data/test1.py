"""
2D single region.

Right triangle with vertices at (0, 0), (1, 0), and (0, 1).
"""

import numpy as np
import os
import sys

path_to_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_to_utils)
from utils import convert_pt_to_polytope, visualize_results

A1 = np.array([[1, 1],
               [-1, 0],
               [0, -1]])
b1 = np.array([1, 0, 0])

s = np.array([0.1, 0.1])
t = np.array([0.4, 0.4])

A_s, b_s = convert_pt_to_polytope(s)
A_t, b_t = convert_pt_to_polytope(t)

As = {"s": A_s, "t": A_t, 0: A1}
bs = {"s": b_s, "t": b_t, 0: b1}

n = A1.shape[1]

# For rounding step: 
N = 1
M = 1

# If file is run directly, visualize the GCS
if __name__ == "__main__":
    visualize_results(As, bs, {**{i: 0 for i in range(len(As)-2)}, **{"s": np.hstack([s,s]), "t": np.hstack([t,t])}}, {**{i: 0 for i in range(len(As)-2)}, **{"s": 1, "t": 1}})