from pydrake.all import (
    MathematicalProgram, 
    Solve, 
)

import numpy as np
import sys
import os
import time

np.set_printoptions(edgeitems=30, linewidth=250, precision=4, suppress=True)

from GCS_utils import *
from utils import *

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test_autogen2 import As, bs, n

V, E, I_v_in, I_v_out = build_graph(As, bs)
print(f"V: {V}")
print(f"E: {E}")

prog = MathematicalProgram()

################################################################################
##### Variable Definitions
################################################################################
SOLVE_CONVEX_RELAXATION = True
# SOLVE_CONVEX_RELAXATION = False

x_v = {}
z_v = {}
y_v = {}
y_e = {}
z_v_e = {}

# Variables for each vertex v ∈ V
for v in V:
    x_v[v] = prog.NewContinuousVariables(2 * n, f'x_{v}')
    z_v[v] = prog.NewContinuousVariables(2 * n, f'z_{v}')
    if SOLVE_CONVEX_RELAXATION:  # Relax Binary variable to 0 <= y_v <= 1
        y_v[v] = prog.NewContinuousVariables(1, f'y_{v}')[0]
        prog.AddBoundingBoxConstraint(0, 1, y_v[v])
    else:
        y_v[v] = prog.NewBinaryVariables(1, f'y_{v}')[0]

# Variables for each edge e ∈ E
for e in E:
    if SOLVE_CONVEX_RELAXATION:  # Relax Binary variable to 0 <= y_e <= 1
        y_e[e] = prog.NewContinuousVariables(1, f'y_e_{e}')[0]
        prog.AddBoundingBoxConstraint(0, 1, y_e[e])
    else:
        y_e[e] = prog.NewBinaryVariables(1, f'y_e_{e}')[0]

# Variables z^e_v for each vertex v ∈ V and each incident edge e ∈ I_v
for v in V:
    for e in I_v_in[v] + I_v_out[v]:
        z_v_e[(v, e)] = prog.NewContinuousVariables(2 * n, f'z_{v}_e_{e}')
        
        
################################################################################
##### Cost
################################################################################
# Path length penalty: sum_{v ∈ V} ||z_v1 - z_v2||^2
for v in V:
    z_v1 = z_v[v][:n]
    z_v2 = z_v[v][n:]
    A = np.hstack([np.eye(z_v1.shape[0]), -np.eye(z_v2.shape[0])])
    b = np.zeros(A.shape[0])
    prog.AddL2NormCost(A, b, np.hstack([z_v1, z_v2]))
    
# Slight penalty for activating edges (to prevent 0-length 2-cycles from happening): sum_{e ∈ E} 1e-4 * y_e
for e in E:
    prog.AddCost(1e-4 * y_e[e])
    

################################################################################
##### Constraints
################################################################################
# Vertex Point Containment Constraints
for v in V:
    m = As[v].shape[0]
    for i in range(2):
        idx = slice(i * n, (i + 1) * n)

        # Constraint 1: A_v z_{v,i} ≤ y_v b_v
        for j in range(m):
            prog.AddConstraint(As[v][j] @ z_v[v][idx] <= y_v[v] * bs[v][j])

        # Constraint 2: A_v (x_{v,i} - z_{v,i}) ≤ (1 - y_v) b_v
        for j in range(m):
            prog.AddConstraint(As[v][j] @ (x_v[v][idx] - z_v[v][idx]) <= (1 - y_v[v]) * bs[v][j])
        
# Edge Point Containment Constraints
for v in V:
    m = As[v].shape[0]
    for e in I_v_in[v] + I_v_out[v]:
        for i in range(2):
            idx = slice(i * n, (i + 1) * n)

            # Constraint 3: A_v z^e_{v,i} ≤ y_e b_v
            for j in range(m):
                prog.AddConstraint(As[v][j] @ z_v_e[(v, e)][idx] <= y_e[e] * bs[v][j])

            # Constraint 4: A_v (x_{v,i} - z^e_{v,i}) ≤ (1 - y_e) b_v
            for j in range(m):
                prog.AddConstraint(As[v][j] @ (x_v[v][idx] - z_v_e[(v, e)][idx]) <= (1 - y_e[e]) * bs[v][j])
            
# Path Continuity Constraints
for e in E:
    v, w = e
    # Constraint 5: z_{v,2}^e = z_{w,1}^e for each edge e = (v, w)
    for d in range(n):  # n because we only check equivalence of one point in each z_v_e (which represents two points)
        prog.AddConstraint(z_v_e[(v, e)][n+d] == z_v_e[(w, e)][d])
    
# Flow Constraints
for v in V:
    delta_sv = delta('s', v)
    delta_tv = delta('t', v)
    
    # Constraint 6: y_v = sum_{e ∈ I_v_in} y_e + δ_{sv} = sum_{e ∈ I_v_out} y_e + δ_{tv}, y_v ≤ 1
    # y_v = sum_{e ∈ I_v_in} y_e + δ_{sv}
    prog.AddConstraint(y_v[v] == sum(y_e[e] for e in I_v_in[v]) + delta_sv)
    # y_v = sum_{e ∈ I_v_out} y_e + δ_{tv}
    prog.AddConstraint(y_v[v] == sum(y_e[e] for e in I_v_out[v]) + delta_tv)
    
# Perspective Flow Constraints
for v in V:
    delta_sv = delta('s', v)
    delta_tv = delta('t', v)
    
    # Constraint 7: z_v = sum_{e ∈ I_v_in} z^e_v + δ_{sv} x_v = sum_{e ∈ I_v_out} z^e_v + δ_{tv} x_v
    for d in range(2*n):   # 2n because z_v is 2n-dimensional
        # z_v = sum_in_z_v_e + δ_{sv} x_v
        prog.AddConstraint(z_v[v][d] == sum(z_v_e[(v, e)][d] for e in I_v_in[v]) + delta_sv * x_v[v][d])
        # z_v = sum_out_z_v_e + δ_{tv} x_v
        prog.AddConstraint(z_v[v][d] == sum(z_v_e[(v, e)][d] for e in I_v_out[v]) + delta_tv * x_v[v][d])
    
################################################################################
##### Solve
################################################################################
print("Beginning MICP Solve.")
start = time.time()
result = Solve(prog)
print(f"Solve Time: {time.time() - start}")
print(f"Solved using: {result.get_solver_id().name()}")

if result.is_success():
    # Solution retreival
    x_v_sol = {}
    z_v_sol = {}
    y_v_sol = {}
    y_e_sol = {}
    z_v_e_sol = {}

    # Variables for each vertex v ∈ V
    for v in V:
        x_v_sol[v] = result.GetSolution(x_v[v])
        z_v_sol[v] = result.GetSolution(z_v[v])
        y_v_sol[v] = result.GetSolution(y_v[v])
        
        # make it more readable
        if np.abs(y_v_sol[v]) < 1e-6:
            y_v_sol[v] = 0
        elif np.abs(y_v_sol[v]) > 1-1e-6:
            y_v_sol[v] = 1

    # Variables for each edge e ∈ E
    for e in E:
        y_e_sol[e] = result.GetSolution(y_e[e])
        
        # make it more readable
        if np.abs(y_e_sol[e]) < 1e-6:
            y_e_sol[e] = 0
        elif np.abs(y_e_sol[e]) > 1-1e-6:
            y_e_sol[e] = 1

    # Variables z^e_v for each vertex v ∈ V and each incident edge e ∈ I_v
    for v in V:
        for e in I_v_in[v] + I_v_out[v]:
            z_v_e_sol[(v, e)] = result.GetSolution(z_v_e[(v, e)])
    
    print(f"Optimal Cost (Path Length): {result.get_optimal_cost()}\n")
    print(f"{x_v_sol=}\n")
    print(f"{y_v_sol=}\n")
    print(f"{y_e_sol=}\n")
    
    if SOLVE_CONVEX_RELAXATION:
        final_cost, x_v_sol, y_v_sol = rounding(y_e_sol, V, E, I_v_out)
        
        print("===============================================================")
        print("POST-ROUNDING")
        print("===============================================================")
        
        print(f"{x_v_sol=}\n")
        print(f"{y_v_sol=}\n")
    
    visualize_results(As, bs, x_v_sol, y_v_sol)
    
else:
    print("solve failed.")
    print(f"{result.get_solution_result()}")
    print(f"{result.GetInfeasibleConstraintNames(prog)}")
    for constraint_binding in result.GetInfeasibleConstraints(prog):
        print(f"{constraint_binding.variables()}")
