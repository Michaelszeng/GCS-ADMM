"""
2D, 2 regions.

Right triangle with vertices at (0, 0), (1, 0), and (0, 1).
Right triangle with vertices at (0.9, 0), (1.9, 0), and (1.9, 1)
"""

import numpy as np

from utils import convert_pt_to_polytope

A1 = np.array([[1, 1],
               [-1, 0],
               [0, -1]])
b1 = np.array([1, 0, 0])

A2 = np.array([
    [0, -1],    # y >= 0
    [1, 0],     # x <= 1.9
    [-1, 1]     # -x + y <= -0.9
])
b2 = np.array([0, 1.9, -0.9])

s = np.array([0, 1])
t = np.array([1.9, 1])

A_s, b_s = convert_pt_to_polytope(s)
A_t, b_t = convert_pt_to_polytope(t)

As = {"s": A_s, "t": A_t, 0: A1, 1: A2}
bs = {"s": b_s, "t": b_t, 0: b1, 1: b2}

n = A1.shape[1]

"""
Solution:

x_v_sol: {0: array([0.  , 1.  , 0.95, 0.05]), 1: array([0.95, 0.05, 1.9 , 1.  ]), 't': array([1.9, 1. , 1.9, 1. ]), 's': array([0., 1., 0., 1.])}
y_v_sol: {0: 1.0, 1: 1.0, 't': 1.0, 's': 1.0}
"""