import numpy as np
import os
import sys

path_to_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_to_utils)
from utils import convert_pt_to_polytope, visualize_results

s = np.array([-2.8886393913556407, 3.145500903658718])

t = np.array([1.8636479068674028, 1.2465548458221605])

A_s, b_s = convert_pt_to_polytope(s, eps=1e-6)
A_t, b_t = convert_pt_to_polytope(t, eps=1e-6)

A0 = np.array([[0.9486832980505139, -0.31622776601683794], [1.0, -0.0], [-1.0, 0.0], [-0.8944271909999159, 0.4472135954999579]])

b0 = np.array([-3.5136418446315263, -2.777777777777771, 3.8888888888888955, 4.223683957499609])

A1 = np.array([[0.0, 1.0], [-0.5144957554275263, -0.8574929257125442], [0.9486832980505139, -0.3162277660168379]])

b1 = np.array([0.555555555555562, 0.9527699174583892, 3.51364184463154])

A2 = np.array([[-0.9486832980505139, 0.31622776601683794], [0.24253562503633308, 0.9701425001453319], [-0.0, -1.0], [0.4472135954999579, -0.8944271909999159]])

b2 = np.array([4.567734398021002, 1.7516461808179695, 0.5555555555555639, -1.2422599874998748])

A3 = np.array([[-0.8944271909999159, -0.4472135954999579], [0.7071067811865476, -0.7071067811865476], [-0.554700196225229, 0.8320502943378437], [-0.0, 1.0]])

b3 = np.array([4.223683957499609, 0.7856742013183928, 0.7704169392017135, 0.555555555555562])

A4 = np.array([[0.0, 1.0], [-0.7071067811865476, -0.7071067811865476], [0.554700196225229, -0.8320502943378436]])

b4 = np.array([1.6666666666666716, 0.7856742013183907, 0.15408338784034611])

A5 = np.array([[0.0, 1.0], [-0.9486832980505138, -0.31622776601683794], [0.7071067811865476, -0.7071067811865476]])

b5 = np.array([1.6666666666666752, 4.216370213557847, -1.5713484026367641])

A6 = np.array([[-0.9701425001453319, 0.24253562503633289], [0.6, 0.7999999999999999], [-0.31622776601683794, -0.9486832980505139], [0.7071067811865476, -0.7071067811865475]])

b6 = np.array([0.4042260417272302, 3.4444444444444526, 0.7027283689263149, 3.1426968052735536])

A7 = np.array([[-0.8944271909999159, -0.4472135954999581], [0.0, -1.0], [0.0, 1.0], [0.5547001962252289, 0.8320502943378437]])

b7 = np.array([0.24845199749998428, -2.77777777777777, 5.000000000000009, 3.235751144647179])

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

# If file is run directly, visualize the GCS
if __name__ == "__main__":
    visualize_results(As, bs, {**{i: 0 for i in range(len(As)-2)}, **{"s": np.hstack([s,s]), "t": np.hstack([t,t])}}, {**{i: 0 for i in range(len(As)-2)}, **{"s": 1, "t": 1}})