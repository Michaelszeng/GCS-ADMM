"""
2D single region.

Right triangle with vertices at (0, 0), (1, 0), and (0, 1).
"""

import numpy as np

from utils import convert_pt_to_polytope

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