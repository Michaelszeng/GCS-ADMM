"""
2D, 4 squares

BENCHMARK 1
"""

import numpy as np
import os
import sys

path_to_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_to_utils)
from utils import convert_pt_to_polytope, visualize_results

# Bottom square
A1 = np.array([[-1, 0],      # x >= 1
               [1, 0],     # x <= 3
                [0, -1],     # y >= 0
               [0, 1]])    # y <= 2
b1 = np.array([-1, 3, 0, 2])

# Left square
A2 = np.array([[-1, 0],      # x >= -0.5
               [1, 0],     # x <= 1.5
                [0, -1],     # y >= 1.5
               [0, 1]])    # y <= 3.5
b2 = np.array([0.5, 1.5, -1.5, 3.5])

# Top square
A3 = np.array([[-1, 0],      # x >= 1
               [1, 0],     # x <= 3
                [0, -1],     # y >= 3
               [0, 1]])    # y <= 5
b3 = np.array([-1, 3, -3, 5])

# Right square
A4 = np.array([[-1, 0],      # x >= 2.5
               [1, 0],     # x <= 4.5
                [0, -1],     # y >= 1.5
               [0, 1]])    # y <= 3.5
b4 = np.array([-2.5, 4.5, -1.5, 3.5])

s = np.array([2, 1])
t = np.array([2, 4])

A_s, b_s = convert_pt_to_polytope(s)
A_t, b_t = convert_pt_to_polytope(t)

As = {"s": A_s, "t": A_t, 0: A1, 1: A2, 2: A3, 3: A4}
bs = {"s": b_s, "t": b_t, 0: b1, 1: b2, 2: b3, 3: b4}

n = A1.shape[1]

# For rounding step: 
N = 2
M = 4

# If file is run directly, visualize the GCS
if __name__ == "__main__":
    visualize_results(As, bs, {**{i: 0 for i in range(len(As)-2)}, **{"s": np.hstack([s,s]), "t": np.hstack([t,t])}}, {**{i: 0 for i in range(len(As)-2)}, **{"s": 1, "t": 1}})