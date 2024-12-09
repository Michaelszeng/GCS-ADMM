"""
8 vertices.
"""

import numpy as np
import os
import sys

path_to_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_to_utils)
from utils import convert_pt_to_polytope, visualize_results

s = np.array([1.893928402271689, -1.7429985950951574])

t = np.array([-0.51, 5.01])

A_s, b_s = convert_pt_to_polytope(s, eps=1e-6)
A_t, b_t = convert_pt_to_polytope(t, eps=1e-6)

A0 = np.array([[0.3262277660168379, 0.9486832980505138], [-1.0, -0.0], [0.8944271909999159, -0.4472135954999579], [0.8320502943378438, -0.554700196225229]])

b0 = np.array([1.3392384683385164, 0.5333333333333414, 2.146625258399806, 2.2188007849009246])

A1 = np.array([[-0.9586832980505138, -0.3162277660168377], [0.0, -1.0], [0.0, 1.0], [0.9486832980505138, 0.3162277660168377]])

b1 = np.array([-0.6846192341692425, -3.733333333333322, 6.933333333333344, 2.698476936677026])

A2 = np.array([[-0.01, -1.0], [1.0, 0.0], [-1.0, 0.0], [-0.9486832980505138, 0.3162277660168379]])

b2 = np.array([2.6766666666666714, 2.6666666666666705, -1.599999999999995, -1.6865480854231305])

A3 = np.array([[-0.7171067811865477, -0.7071067811865475], [0.7071067811865479, -0.7071067811865472], [0.0, 1.0], [0.4472135954999579, 0.8944271909999159]])

b3 = np.array([-0.7642472332656399, -3.0169889330625916, 5.866666666666677, 4.531764434399584])

A4 = np.array([[-0.01, -1.0], [-0.9805806756909201, 0.19611613513818396], [0.7808688094430304, 0.6246950475544242]])

b4 = np.array([-0.5433333333333231, 3.7654297946531434, 1.582560787137884])

A5 = np.array([[-0.3262277660168379, 0.9486832980505138], [-1.0, -0.0], [1.0, 0.0], [0.5547001962252294, -0.8320502943378436]])

b5 = np.array([-0.32730961708462054, 0.5333333333333397, 2.6666666666666723, 2.8104809942078335])

A6 = np.array([[-0.8220502943378437, 0.5547001962252288], [0.7071067811865475, 0.7071067811865475], [-0.0, -1.0], [0.8944271909999161, -0.44721359549995765]])

b6 = np.array([2.5246408895543817, 6.033977866125216, -3.733333333333321, -0.23851391759996451])

A7 = np.array([[-0.01, -1.0], [-0.4472135954999579, 0.8944271909999159], [0.7071067811865475, 0.7071067811865475], [0.44721359549995765, 0.8944271909999161]])

b7 = np.array([-1.5899999999999879, 1.6695974231998543, 6.033977866125217, 5.00879226959954])

As = {
    "s": A_s,
    "t": A_t,
    0: A0,
    1: A1,
    2: A2,
    3: A3,
    4: A4,
    5: A5,
    6: A6,
    7: A7,
}

bs = {
    "s": b_s,
    "t": b_t,
    0: b0,
    1: b1,
    2: b2,
    3: b3,
    4: b4,
    5: b5,
    6: b6,
    7: b7,
}

n = A0.shape[1]
# For rounding step: 
N = 2
M = 3

# If file is run directly, visualize the GCS
if __name__ == "__main__":
   visualize_results(As, bs, {**{i: 0 for i in range(len(As)-2)}, **{"s": np.hstack([s,s]), "t": np.hstack([t,t])}}, {**{i: 0 for i in range(len(As)-2)}, **{"s": 1, "t": 1}})