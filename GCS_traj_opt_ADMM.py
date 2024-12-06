from pydrake.all import (
    MathematicalProgram, 
    Solve, 
    Constraint,
)

import numpy as np
import pandas as pd
import sys
import os
import time

np.set_printoptions(edgeitems=30, linewidth=250, precision=4, suppress=True)

from utils import *

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test3 import As, bs, n

V, E, I_v_in, I_v_out = build_graph(As, bs)
print(f"V: {V}")
print(f"E: {E}")

# Variables to store current global values of split x and z variables
#                      x_v          z_v         y_v
x_global = np.zeros(2*n*len(V) + 2*n*len*(V) + len(V))
#                                       x_v_e                                            z_v_e                             y_e
z_global = np.zeros(sum(2*n*len((I_v_in[v] + I_v_out[v])) for v in V), sum(2*n*len((I_v_in[v] + I_v_out[v])) for v in V), len(E))

# Public variable set to use to express consensus constraints and penalties
x_v = {}
z_v = {}
y_v = {}
x_v_e = {}
z_v_e = {}
y_e = {}
        
def build_A_B_c_consensus_matrices(x_v, z_v, y_v, x_v_e, z_v_e, y_e):
    """
    A function designed to take advantage of Drake's built-in optimization 
    tools to more easily express the consensus constraints and penalties in 
    standard Ax + Bz = c form. Builds and returns the A, B, and c matrices 
    for the consensus constraints for the given problem.
    
    Args:
        Empty dictionaries for all variables in the split ADMM formulation.

    Returns:
        np.ndarray: A matrix.
        np.ndarray: B matrix.
        float: c vector.
    """
    def is_x_var(var):
        return prog.FindDecisionVariableIndex(var) < z_idx
    
    # First, Build the MathematicalProgram containing all the variables in the split ADMM formulation.
    prog = MathematicalProgram()
    
    # Build variable set for x variables
    for v in V:
        x_v[v] = prog.NewContinuousVariables(2 * n, f'x_{v}')
        z_v[v] = prog.NewContinuousVariables(2 * n, f'z_{v}')
        y_v[v] = prog.NewBinaryVariables(1, f'y_{v}')[0]
    
    # Build variable set for z variables
    first_z_var = None
    for v in V:
        for e in I_v_in[v] + I_v_out[v]:
            if first_z_var is None:
                first_z_var = x_v_e[(v, e)]
            x_v_e[(v, e)] = prog.NewContinuousVariables(2 * n, f'x_{v}_e_{e}')
            z_v_e[(v, e)] = prog.NewContinuousVariables(2 * n, f'z_{v}_e_{e}')
    for e in E:
        y_e[e] = prog.NewBinaryVariables(1, f'y_e_{e}')[0]
        
    # Make it easier to index into the z variables
    z_idx = prog.FindDecisionVariableIndex(first_z_var)
    
    
    # Then, encode all the consensus constraints in the MathematicalProgram
    consensus_costraints = []  # list of all consensus constraints
    for e in E:
        # Vertex-Edge consensus constraints
        v, w = e
        # x_v = x_v_e
        consensus_costraints.append(x_v_e[(v, e)] == x_v[v])
        # x_w = x_w_e
        consensus_costraints.append(x_v_e[(w, e)] == x_v[w])
        
    for v in V:
        # Flow and Perspective Flow consensus constraints
        delta_sv = delta('s', v)
        delta_tv = delta('t', v)

        # y_v = sum_{e ∈ I_v_in} y_e + δ_{sv}
        consensus_costraints.append(y_v[v] == sum(y_e[e] for e in I_v_in[v]) + delta_sv)
        # y_v = sum_{e ∈ I_v_out} y_e + δ_{tv}
        consensus_costraints.append(y_v[v] == sum(y_e[e] for e in I_v_out[v]) + delta_tv)
        
        for d in range(2*n):   # 2n because z_v is 2n-dimensional
            # z_v = sum_in_z_v_e + δ_{sv} x_v
            consensus_costraints.append(z_v[v][d] == sum(z_v_e[(v, e)][d] for e in I_v_in[v]) + delta_sv * x_v[v][d])
            # z_v = sum_out_z_v_e + δ_{tv} x_v
            consensus_costraints.append(z_v[v][d] == sum(z_v_e[(v, e)][d] for e in I_v_out[v]) + delta_tv * x_v[v][d])
            
            
    # Now, construct A, B, c
    A = np.zeros((len(consensus_costraints), prog.num_vars()))
    B = np.zeros((len(consensus_costraints), z.num_vars()))
    c = np.zeros(len(consensus_costraints))
    for i, formula in enumerate(consensus_costraints):
        constraint_binding = prog.AddConstraint(formula)
        constraint = constraint_binding.evaluator()  # Must be linear constraint
        constraint_vars = constraint_binding.variables()
        
        # We assume the linear (equality) constraint given only has one row (i.e. only encodes a scalar equality constraint)
        constraint_A = constraint.GetDenseA()[0]
        constraint_b = constraint.upper_bound()[0]
        assert constraint.lower_bound() == constraint.upper_bound()

        for j, var in enumerate(constraint_vars):
            var_idx = prog.FindDecisionVariableIndex(var)
            
            if is_x_var(var):
                A[i, var_idx] += constraint_A[j]
            else:  # is z var
                B[i, var_idx - z_idx] += constraint_A[j]
            
            c[i] = constraint_b
            
    return A, B, c

A, B, c = build_A_B_c_consensus_matrices(x_v, z_v, y_v, x_v_e, z_v_e, y_e)


def vertex_update(rho, v, x_v_e, z_v_e, y_e, mu):
    """
    Perform vertex update ("x-update") step for a single vertex v.
    
    Args:
        rho: scalar penalty parameter.
        x_v_e: list of numpy arrays representing the current values of x_v_e for all edges ∈ I_v.
        z_v_e: list of numpy arrays representing the current values of z_v_e for all edges ∈ I_v.
        y_e: list of binary variables representing the current values of y_e for all edges ∈ I_v.
    """
    prog = MathematicalProgram()
    x_v = prog.NewContinuousVariables(2 * n, f'x_v')
    z_v = prog.NewContinuousVariables(2 * n, f'z_v')
    y_v = prog.NewBinaryVariables(1, f'y_v')[0]
    
    # Path length penalty: ||z_v1 - z_v2||^2
    z_v1 = z_v[:n]
    z_v2 = z_v[n:]
    A = np.hstack([np.eye(z_v1.shape[0]), -np.eye(z_v2.shape[0])])
    b = np.zeros(A.shape[0])
    prog.AddL2NormCost(A, b, np.hstack([z_v1, z_v2]))
    
    # Concensus Constraint Penalty: (rho/2) * ||Ax + Bz + mu||^2
    # Define x vector that contain fixed valus for all but the variables corresponding to v
    
    prog.AddCost((rho/2) * (A @ x + B @ z_global - c + mu).T @ (A @ x + B @ z_global - c + mu))
    
    
    prog.AddCost((rho/2) * ((sum(x_v - x_v_e[i] for i in range(len(x_v_e))) + mu).T @ (sum(x_v - x_v_e[i] for i in range(len(x_v_e))) + mu)))
    
    # Point containment costraints
    m = As[v].shape[0]
    for i in range(2):
        idx = slice(i * n, (i + 1) * n)

        # Constraint 1: A_v z_{v,i} ≤ y_v b_v
        for j in range(m):
            prog.AddConstraint(As[v][j] @ z_v[v][idx] <= y_v[v] * bs[v][j])
            
        # Constraint 2: A_v (x_{v,i} - z_{v,i}) ≤ (1 - y_v) b_v
        for j in range(m):
            prog.AddConstraint(As[v][j] @ (x_v[v][idx] - z_v[v][idx]) <= (1 - y_v[v]) * bs[v][j])
            
    result = Solve(prog)
    
    if result.is_success():
        # Solution retreival
        x_v_sol = result.GetSolution(x_v)
        z_v_sol = result.GetSolution(z_v)
        y_v_sol = result.GetSolution(y_v)

        # print(f"{x_v_sol=}\n")
        # print(f"{z_v_sol=}\n")
        # print(f"{y_v_sol=}\n")
        
        return x_v_sol, z_v_sol, y_v_sol
    else:
        print("solve failed.")
        print(f"{result.get_solution_result()}")
        print(f"{result.GetInfeasibleConstraintNames(prog)}")
        for constraint_binding in result.GetInfeasibleConstraints(prog):
            print(f"{constraint_binding.variables()}")


def edge_update(rho, z_v, y_v, mu):
    prog = MathematicalProgram()
    prog.AddCost()


def get_next_vertex_values(rho, xe_k, mu):
    constraints = u.constraints + v.constraints
    cost = u.cost + v.cost + rho / 2 * cp.sum_squares(xe - xe_k + mu)
    prog = cp.Problem(cp.Minimize(cost), constraints)
    prog.solve()
    assert prog.status == "optimal"
    return xu.value, xv.value


def get_next_edge_values(rho, xv_e_k, mu):
    constraints = e.constraints
    cost = e.cost + rho / 2 * cp.sum_squares(xv_e_k - xe + mu)
    prog = cp.Problem(cp.Minimize(cost), constraints)
    prog.solve()
    assert prog.status == "optimal"
    return xe.value


def get_next_consensus_var(xe_vertex, xe_edge, mu):
    return mu + xe_vertex - xe_edge