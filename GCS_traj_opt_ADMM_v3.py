"""
"Full vertex splitting"

Note on variable naming conventions: variable z_{e_u^w} is named z_e_u_w
"""

from pydrake.all import (
    MathematicalProgram, 
    MosekSolver,
    SolveInParallel,
    Variable,
)

import numpy as np
import pandas as pd
import sys
import os
import time
import re

np.set_printoptions(edgeitems=30, linewidth=250, precision=4, suppress=True)

from utils import *

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)
from test4 import As, bs, n

V, E, I_v_in, I_v_out = build_graph(As, bs)
print(f"V: {V}")
print(f"E: {E}")

# Establish solver
mosek_solver = MosekSolver()
if not mosek_solver.available():
    print("WARNING: MOSEK unavailable.")

class ConsensusManager():
    """
    A class designed to take advantage of Drake's built-in optimization tools to
    more easily express the consensus constraints and penalties in standard
    Ax + Bz = c form.
    """
    def __init__(self):
        """
        Builds the MathematicalProgram containing all the variables in the split
        ADMM formulation.
        
        Args:
            Empty dictionaries for all variables in the split ADMM formulation.
        """
        # x variables
        self.x_v = {}  # key: v
        self.z_v = {}  # key: v
        self.y_v = {}  # key: v
        self.z_e_u_v = {}  # key: (e, u, v) - e is the edge; u is the vertex that gives z_e_u_v its value (i.e. z_e_u_v = y_u x_u); u is either v or the other vertex in the edge; v is the current vertex
        self.y_e_v = {}  # key: (e, v) - e is the edge; v is the current vertex
        
        # z variables
        self.z_e_u_e = {}  # key: (e, v) - e is the edge, v is the one of the vertices in the edge
        self.y_e_e = {}  # key: e

        self.prog = MathematicalProgram()
        
        # Build variable set for x variables
        x_v_start_idx = self.prog.num_vars()
        for v in V:
            self.x_v[v] = self.prog.NewContinuousVariables(2 * n, f'x_{v}')
        self.x_v_indices_in_x = slice(x_v_start_idx, self.prog.num_vars())  # Keep track of x_v indices in x variable set
        
        z_v_start_idx = self.prog.num_vars()
        for v in V:
            self.z_v[v] = self.prog.NewContinuousVariables(2 * n, f'z_{v}')
        self.z_v_indices_in_x = slice(z_v_start_idx, self.prog.num_vars())  # Keep track of z_v indices in x variable set
        
        y_v_start_idx = self.prog.num_vars()
        for v in V:
            self.y_v[v] = self.prog.NewBinaryVariables(1, f'y_{v}')[0]
        self.y_v_indices_in_x = slice(y_v_start_idx, self.prog.num_vars())  # Keep track of y_v indices in x variable set
        
        z_e_u_v_start_idx = self.prog.num_vars()
        for v in V:
            for e in I_v_in[v] + I_v_out[v]:
                self.z_e_u_v[(e, e[0], v)] = self.prog.NewContinuousVariables(2 * n, f'z_{e}_{e[0]}_{v}')
                self.z_e_u_v[(e, e[1], v)] = self.prog.NewContinuousVariables(2 * n, f'z_{e}_{e[1]}_{v}')
        self.z_e_u_v_indices_in_x = slice(z_e_u_v_start_idx, self.prog.num_vars())  # Keep track of z_e_u_v indices in x variable set
        
        y_e_v_start_idx = self.prog.num_vars()
        for v in V:
            for e in I_v_in[v] + I_v_out[v]:
                self.y_e_v[(e, v)] = self.prog.NewBinaryVariables(1, f'y_{e}_{v}')[0]
        self.y_e_v_indices_in_x = slice(y_e_v_start_idx, self.prog.num_vars())  # Keep track of y_e_v indices in x variable set
        
        # Build variable set for z variables
        first_z_var = None
        z_e_u_e_start_idx = 0
        for e in E:
            self.z_e_u_e[(e, e[0])] = self.prog.NewContinuousVariables(2 * n, f'z_{e}_{e[0]}_e')  # extra e at the end to differentiate this as an edge variable (i.e. part of z variable set)
            self.z_e_u_e[(e, e[1])] = self.prog.NewContinuousVariables(2 * n, f'z_{e}_{e[1]}_e')
            if first_z_var is None:
                first_z_var = self.z_e_u_e[(e, e[0])][0]
                # Make it easier to index into the z variables
                self.z_idx = self.prog.FindDecisionVariableIndex(first_z_var)
        self.z_e_u_e_indices_in_z = slice(z_e_u_e_start_idx, self.prog.num_vars() - self.z_idx)  # Keep track of z_e_u_e indices in z variable set
            
        y_e_e_start_idx = self.prog.num_vars() - self.z_idx
        for e in E:
            self.y_e_e[e] = self.prog.NewBinaryVariables(1, f'y_{e}_e')[0]  # extra e at the end to differentiate this as an edge variable (i.e. part of z variable set)
        self.y_e_e_indices_in_z = slice(y_e_e_start_idx, self.prog.num_vars() - self.z_idx)  # Keep track of y_e indices in z variable set
        
        self.A = None
        self.B = None
        self.c = None
        
    def is_x_var(self, var):
        return self.prog.FindDecisionVariableIndex(var) < self.z_idx
        
    def build_A_B_c_consensus_matrices(self):
        """
        Builds the A, B, and c matrices for the consensus constraints for the 
        given problem.

        Returns:
            np.ndarray: A matrix.
            np.ndarray: B matrix.
            float: c vector.
        """
        # First, encode all the consensus constraints in the MathematicalProgram
        consensus_costraints = []  # list of all consensus constraints
        for e in E:
            u, w = e
            
            for dim in range(n):
                # z_e_u_e = z_e_u_u = z_e_u_w
                consensus_costraints.append(self.z_e_u_e[(e, u)][dim] == self.z_e_u_v[(e, u, u)][dim])
                consensus_costraints.append(self.z_e_u_e[(e, u)][dim] == self.z_e_u_v[(e, u, w)][dim])
                # z_e_w_e = z_e_w_w = z_e_w_u
                consensus_costraints.append(self.z_e_u_e[(e, w)][dim] == self.z_e_u_v[(e, w, w)][dim])
                consensus_costraints.append(self.z_e_u_e[(e, w)][dim] == self.z_e_u_v[(e, w, u)][dim])
            
            # y_e_e = y_e_u = y_e_w
            consensus_costraints.append(self.y_e_e[e] == self.y_e_v[(e, u)])
            consensus_costraints.append(self.y_e_e[e] == self.y_e_v[(e, w)])
                
        # Now, construct A, B, c
        A = np.zeros((len(consensus_costraints), self.z_idx))
        B = np.zeros((len(consensus_costraints), self.prog.num_vars() - self.z_idx))
        c = np.zeros(len(consensus_costraints))
        for i, formula in enumerate(consensus_costraints):
            constraint_binding = self.prog.AddConstraint(formula)
            constraint = constraint_binding.evaluator()  # Must be linear constraint
            constraint_vars = constraint_binding.variables()
            
            # We assume the linear (equality) constraint given only has one row (i.e. only encodes a scalar equality constraint)
            constraint_A = constraint.GetDenseA()[0]
            constraint_b = constraint.upper_bound()[0]
            assert constraint.lower_bound() == constraint.upper_bound()

            for j, var in enumerate(constraint_vars):
                var_idx = self.prog.FindDecisionVariableIndex(var)
                
                if self.is_x_var(var):
                    A[i, var_idx] += constraint_A[j]
                else:  # is z var
                    B[i, var_idx - self.z_idx] += constraint_A[j]
                
                c[i] = constraint_b
                
        # Store for internal use later
        self.A = A
        self.B = B
        self.c = c
                
        return A, B, c
    
    def get_x_var_indices(self, vars, v, e=None, u=None):
        """
        Helper funtion for each vertex update to find the index of the given
        variable in the x variable set.
        
        Args:
            vars: np.ndarray, variable array for the variables whose indices in the x variable set are being searched for.
            v: vertex key for the variable/vertex update.
            e: edge key
            u: vertex key
            
        Returns:
            slice: slice object representing the range of indices in the x variable set corresponding to the given variable.
        """
        if isinstance(vars, np.ndarray):
            vars = vars[0]
            
        if re.search("x_.*", vars.get_name()):
            var = self.x_v[v]
        # Search for z_e_u_v first because it has the most specific naming convention
        elif re.search("z_\(.*\).*", vars.get_name()):
            var = self.z_e_u_v[(e, u, v)]
        elif re.search("y_\(.*\).*", vars.get_name()):
            var = [self.y_e_v[(e, v)]]
        elif re.search("z_.*", vars.get_name()):
            var = self.z_v[v]
        elif re.search("y_.*", vars.get_name()):
            var = [self.y_v[v]]
        else:
            raise ValueError("Invalid variable name.")
        
        idx_first = self.prog.FindDecisionVariableIndex(var[0])
        idx_last = self.prog.FindDecisionVariableIndex(var[-1])
        assert idx_first < self.z_idx and idx_last < self.z_idx  # Ensure the variable is in the x variable set
        return slice(idx_first, idx_last+1)
    
    def get_z_var_indices(self, vars, e, v=None):
        """
        Helper funtion for each edge update to find the index of the given
        variable in the z variable set.
        
        Args:
            vars: np.ndarray, variable array for the variables whose indices in the x variable set are being searched for.
            v: vertex key for the variable/edge update. (Note: not applicable for y_e)
            e: edge key for the variable/edge update.
            
        Returns:
            slice: slice object representing the range of indices in the z variable set corresponding to the given variable.
        """
        if isinstance(vars, np.ndarray):
            vars = vars[0]
        
        if re.search("z_.*_e", vars.get_name()):
            var = self.z_e_u_e[(e, v)]
        elif re.search("y_.*_e", vars.get_name()):
            var = [self.y_e_e[e]]
        else:
            raise ValueError("Invalid variable name.")
        
        idx_first = self.prog.FindDecisionVariableIndex(var[0]) - self.z_idx
        idx_last = self.prog.FindDecisionVariableIndex(var[-1]) - self.z_idx
        assert idx_first >= 0 and idx_last >= 0  # Ensure the variable is in the z variable set
        return slice(idx_first, idx_last+1)
    
    def get_x_v_var_indices_in_x(self):
        """
        Retrieve the index slice for all the x_v variables within the x variable set.
        
        Returns:
            slice: slice object representing the range of indices in the x variable set corresponding to the x_v variables.
        """
        return self.x_v_indices_in_x
    
    def get_z_v_var_indices_in_x(self):
        """
        Retrieve the index slice for all the z_v variables within the x variable set.
        
        Returns:
            slice: slice object representing the range of indices in the x variable set corresponding to the z_v variables.
        """
        return self.z_v_indices_in_x
    
    def get_y_v_var_indices_in_x(self):
        """
        Retrieve the index slice for all the y_v variables within the x variable set.
        
        Returns:
            slice: slice object representing the range of indices in the x variable set corresponding to the y_v variables.
        """
        return self.y_v_indices_in_x
    
    def get_z_e_u_v_var_indices_in_x(self):
        """
        Retrieve the index slice for all the z_e_u_v variables within the x variable set.
        
        Returns:
            slice: slice object representing the range of indices in the x variable set corresponding to the z_e_u_v variables.
        """
        return self.z_e_u_v_indices_in_x
    
    def get_y_e_v_var_indices_in_x(self):
        """
        Retrieve the index slice for all the y_e_v variables within the x variable set.
        
        Returns:
            slice: slice object representing the range of indices in the x variable set corresponding to the y_e_v variables.
        """
        return self.y_e_v_indices_in_x
    
    def get_z_e_u_e_var_indices_in_z(self):
        """
        Retrieve the index slice for all the z_e_u_e variables within the z variable set.
        
        Returns:
            slice: slice object representing the range of indices in the z variable set corresponding to the z_e_u_e variables.
        """
        return self.z_e_u_e_indices_in_z
    
    def get_y_e_e_var_indices_in_z(self):
        """
        Retrieve the index slice for all the y_e_e variables within the z variable set.
        
        Returns:
            slice: slice object representing the range of indices in the z variable set corresponding to the y_e_e variables.
        """
        return self.y_e_e_indices_in_z
    
    def get_num_x_vars(self):
        return self.z_idx
    
    def get_num_z_vars(self):
        return self.prog.num_vars() - self.z_idx
    
    def get_num_mu_vars(self):
        if self.A is None:
            raise ValueError("Call build_A_B_c_consensus_matrices() first.")
        return self.A.shape[0]  # Number of rows of A = number of consensus constraints = number of mu variables
    
    
# Build consensus manager to handle construction of A, B, c matrices for consensus constraints and penalties
consensus_manager = ConsensusManager()
A, B, c = consensus_manager.build_A_B_c_consensus_matrices()  # Just build these once; they remain constant throughout optimization

# Variables to store current global values of split x and z variables at current time step
# Consists of x_v, z_v, y_v, z_e_u_v, y_e_v
x_global = np.zeros(consensus_manager.get_num_x_vars())
# Consists of z_e_u_e, y_e_e
z_global = np.zeros(consensus_manager.get_num_z_vars())

mu_global = np.zeros(consensus_manager.get_num_mu_vars())


def vertex_update(rho, v):
    """
    Perform vertex update ("x-update") step for a single vertex v.
    
    Args:
        rho: scalar penalty parameter.
        v: vertex key for the vertex being updated.
    """  
    prog = MathematicalProgram()
    
    # Variable Definitions
    x_v = prog.NewContinuousVariables(2 * n, f'x_v')
    z_v = prog.NewContinuousVariables(2 * n, f'z_v')
    y_v = prog.NewContinuousVariables(1, f'y_v')[0]  # Relax y_v to 0 <= y_v <= 1
    prog.AddBoundingBoxConstraint(0, 1, y_v)
    
    z_e_u_v = {}
    y_e_v = {}
    
    for e in I_v_in[v] + I_v_out[v]:
        z_e_u_v[(e, e[0])] = prog.NewContinuousVariables(2 * n, f'z_{e}_{e[0]}_v')
        z_e_u_v[(e, e[1])] = prog.NewContinuousVariables(2 * n, f'z_{e}_{e[1]}_v')

    for e in I_v_in[v] + I_v_out[v]:
        y_e_v[e] = prog.NewContinuousVariables(1, f'y_{e}_v')[0]  # Relax y_e_v to 0 <= y_v <= 1
        prog.AddBoundingBoxConstraint(0, 1, y_e_v[e])
        
    # Path Length Penalty: ||z_v1 - z_v2||^2
    z_v1 = z_v[:n]
    z_v2 = z_v[n:]
    A_path_len_penalty = np.hstack([np.eye(z_v1.shape[0]), -np.eye(z_v2.shape[0])])
    b_path_len_penalty = np.zeros(A_path_len_penalty.shape[0])
    prog.AddL2NormCost(A_path_len_penalty, b_path_len_penalty, np.hstack([z_v1, z_v2]))
    
    # Edge Activation Penalty: sum_{e ∈ I_v} eps * y_e
    for e in I_v_in[v] + I_v_out[v]:
        prog.AddLinearCost(1e-4 * y_e_v[e])
    
    # Concensus Constraint Penalty: (rho/2) * ||Ax + Bz + mu||^2
    # Define x vectors containing fixed values and variable values
    var_indices = np.concatenate([np.arange(s.start, s.stop) for s in [consensus_manager.get_x_var_indices(x_v, v), 
                                                                       consensus_manager.get_x_var_indices(z_v, v), 
                                                                       consensus_manager.get_x_var_indices(y_v, v)] + 
                                                                      [consensus_manager.get_x_var_indices(z_e_u_v[(e, e[0])], v, e, e[0]) for e in I_v_in[v] + I_v_out[v]] +
                                                                      [consensus_manager.get_x_var_indices(z_e_u_v[(e, e[1])], v, e, e[1]) for e in I_v_in[v] + I_v_out[v]] +
                                                                      [consensus_manager.get_x_var_indices(y_e_v[e], v, e) for e in I_v_in[v] + I_v_out[v]]])
    x_fixed = np.delete(x_global.copy(), var_indices)
    x_var = np.concatenate([x_v, z_v, [y_v], *[z_e_u_v[(e, e[0])] for e in I_v_in[v] + I_v_out[v]], 
                                             *[z_e_u_v[(e, e[1])] for e in I_v_in[v] + I_v_out[v]], 
                                             *[[y_e_v[e]] for e in I_v_in[v] + I_v_out[v]]])
    # Split A into fixed part and variable part. Note that this is necessary simply because we can't stack fixed and variable parts of x into a single np vector.
    A_fixed = np.delete(A, var_indices, axis=1)
    A_var = A[:, var_indices]
    # z and mu are all fixed
    # print(f"A_var.shape: {A_var.shape}")
    # print(f"x_var.shape: {x_var.shape}")
    # print(f"B.shape: {B.shape}")
    # print(f"z_global.shape: {z_global.shape}")
    # print(f"c.shape: {c.shape}")
    residual = A_fixed @ x_fixed + A_var @ x_var + B @ z_global - c
    # residual = A_var @ x_var + B @ z_global - c  # NOTE: DOES NOT SEEM TO MATTER WHETHER WE INCLUDE A_fixed @ x_fixed OR NOT
    prog.AddCost((rho/2) * (residual + mu_global).T @ (residual + mu_global))
    
    # Point Containment Constraints
    m = As[v].shape[0]
    for i in range(2):
        idx = slice(i * n, (i + 1) * n)

        # Constraint 1: A_v z_{v,i} ≤ y_v b_v
        for j in range(m):
            prog.AddConstraint(As[v][j] @ z_v[idx] <= y_v * bs[v][j])
            
        # Constraint 2: A_v (x_{v,i} - z_{v,i}) ≤ (1 - y_v) b_v
        for j in range(m):
            prog.AddConstraint(As[v][j] @ (x_v[idx] - z_v[idx]) <= (1 - y_v) * bs[v][j])
            
    # Edge Point Containment Constraints
    m = As[v].shape[0]
    for e in I_v_in[v] + I_v_out[v]:
        for i in range(2):
            idx = slice(i * n, (i + 1) * n)

            # Constraint 3: A_v z_{e_v^v,i} ≤ y_e^v b_v
            for j in range(m):
                prog.AddConstraint(As[v][j] @ z_e_u_v[(e, v)][idx] <= y_e_v[e] * bs[v][j])

            # Constraint 4: A_v (x_{v,i} - z_{e_v^v,i}) ≤ (1 - y_e^v) b_v
            for j in range(m):
                prog.AddConstraint(As[v][j] @ (x_v[idx] - z_e_u_v[(e, v)][idx]) <= (1 - y_e_v[e]) * bs[v][j])
                
    # Path Continuity Constraints
    for e in I_v_in[v] + I_v_out[v]:
        u, w = e
        # Constraint 5: z_{e_u^v,2} = z_{e_w^v,1} for each edge e = (u, w)
        for d in range(n):  # n because we only check equivalence of one point in each z_v_e (which represents two points)
            prog.AddConstraint(z_e_u_v[(e, u)][n+d] == z_e_u_v[(e, w)][d])
            
    # Flow Constraints
    delta_sv = delta('s', v)
    delta_tv = delta('t', v)
    # Constraint 6: y_v = sum_{e ∈ I_v_in} y_e^v + δ_{sv} = sum_{e ∈ I_v_out} y_e^v + δ_{tv}, y_v ≤ 1
    # y_v = sum_{e ∈ I_v_in} y_e^v + δ_{sv}
    prog.AddConstraint(y_v == sum(y_e_v[e] for e in I_v_in[v]) + delta_sv)
    # y_v = sum_{e ∈ I_v_out} y_e^v + δ_{tv}
    prog.AddConstraint(y_v == sum(y_e_v[e] for e in I_v_out[v]) + delta_tv)
    
    # Perspective Flow Constraints  
    # Constraint 7: z_v = sum_{e ∈ I_v_in} z_{e_v^v} + δ_{sv} x_v = sum_{e ∈ I_v_out} z_{e_v^v} + δ_{tv} x_v
    for d in range(2*n):   # 2n because z_v is 2n-dimensional
        # z_v = sum_{e ∈ I_v_in} z_{e_v^v} + δ_{sv} x_v
        prog.AddConstraint(z_v[d] == sum(z_e_u_v[(e, v)][d] for e in I_v_in[v]) + delta_sv * x_v[d])
        # z_v = sum_{e ∈ I_v_out} z_{e_v^v} + δ_{tv} x_v
        prog.AddConstraint(z_v[d] == sum(z_e_u_v[(e, v)][d] for e in I_v_out[v]) + delta_tv * x_v[d])

    return prog, x_v, z_v, y_v, z_e_u_v, y_e_v


def parallel_vertex_update(rho):
    """
    Solve vertex updates in parallel. 
    
    Args:
        rho: scalar penalty parameter.
        
    Returns:
        np.ndarray: updated x variable set.
        float: elapsed solve time.
    """
    # Accumulate all vertex update programs
    progs = []
    prog_vars = []
    for v in V:
        prog, x_v, z_v, y_v, z_e_u_v, y_e_v = vertex_update(rho, v)
        progs.append(prog)
        prog_vars.append((x_v, z_v, y_v, z_e_u_v, y_e_v))
        
    # Solve all vertex update programs in parallel
    t_start = time.time()
    results = SolveInParallel(progs, solver_ids=[MosekSolver().solver_id()] * len(progs))
    t_elapsed = time.time() - t_start
    x_updated = np.zeros(consensus_manager.get_num_x_vars())
    for i, result in enumerate(results):
        # Define for convenience
        x_v = prog_vars[i][0]
        z_v = prog_vars[i][1]
        y_v = prog_vars[i][2]
        z_e_u_v = prog_vars[i][3]
        y_e_v = prog_vars[i][4]
        
        if result.is_success():
            # Solution retreival
            v = V[i]
            x_v_sol = result.GetSolution(x_v)
            z_v_sol = result.GetSolution(z_v)
            y_v_sol = result.GetSolution(y_v)
            z_e_u_v_sol = {}
            y_e_v_sol = {}
            
            for e in I_v_in[v] + I_v_out[v]:
                z_e_u_v_sol[(e, e[0])] = result.GetSolution(z_e_u_v[(e, e[0])])
                z_e_u_v_sol[(e, e[1])] = result.GetSolution(z_e_u_v[(e, e[1])])
                y_e_v_sol[e] = result.GetSolution(y_e_v[e])

            # Build next x value set
            x_updated[consensus_manager.get_x_var_indices(x_v, v)] = x_v_sol
            x_updated[consensus_manager.get_x_var_indices(z_v, v)] = z_v_sol
            x_updated[consensus_manager.get_x_var_indices(y_v, v)] = y_v_sol
            for e in I_v_in[v] + I_v_out[v]:
                x_updated[consensus_manager.get_x_var_indices(z_e_u_v[(e, e[0])], v, e, e[0])] = z_e_u_v_sol[(e, e[0])]
                x_updated[consensus_manager.get_x_var_indices(z_e_u_v[(e, e[1])], v, e, e[1])] = z_e_u_v_sol[(e, e[1])]
                x_updated[consensus_manager.get_x_var_indices(y_e_v[e], v, e)] = y_e_v_sol[(e)]
            
        else:
            print("solve failed.")
            print(f"{result.get_solution_result()}")
            print(f"{result.GetInfeasibleConstraintNames(prog)}")
            for constraint_binding in result.GetInfeasibleConstraints(prog):
                print(f"{constraint_binding.variables()}")
            
            # Reuse old x values
            x_updated[consensus_manager.get_x_var_indices(x_v, v)] = x_global[consensus_manager.get_x_var_indices(x_v, v)]
            x_updated[consensus_manager.get_x_var_indices(z_v, v)] = x_global[consensus_manager.get_x_var_indices(z_v, v)]
            x_updated[consensus_manager.get_x_var_indices(y_v, v)] = x_global[consensus_manager.get_x_var_indices(y_v, v)]
            for e in I_v_in[v] + I_v_out[v]:
                x_updated[consensus_manager.get_x_var_indices(z_e_u_v, v, e, e[0])] = x_global[consensus_manager.get_x_var_indices(z_e_u_v, v, e, e[0])]
                x_updated[consensus_manager.get_x_var_indices(z_e_u_v, v, e, e[1])] = x_global[consensus_manager.get_x_var_indices(z_e_u_v, v, e, e[1])]
                x_updated[consensus_manager.get_x_var_indices(y_v_sol, v, e)] = x_global[consensus_manager.get_x_var_indices(y_v_sol, v, e)]
                
    return x_updated, t_elapsed


def edge_update(rho, e):
    """
    Perform edge update ("z-update") step for a single vertex edge e = (u,w).
    
    Args:
        rho: scalar penalty parameter.
        e: edge key for the edge being updated.
    """
    u, w = e
    
    z_e_u_e = (1/2) * (x_global[consensus_manager.get_x_var_indices(Variable(f"z_{e}_{u}_{u}"), u, e, u)] + 
                       x_global[consensus_manager.get_x_var_indices(Variable(f"z_{e}_{u}_{w}"), w, e, u)])
    
    z_e_w_e = (1/2) * (x_global[consensus_manager.get_x_var_indices(Variable(f"z_{e}_{w}_{w}"), w, e, w)] + 
                       x_global[consensus_manager.get_x_var_indices(Variable(f"z_{e}_{w}_{u}"), u, e, w)])
    
    y_e_e = (1/2) * (x_global[consensus_manager.get_x_var_indices(Variable(f"y_{e}_{u}"), u, e)] + 
                     x_global[consensus_manager.get_x_var_indices(Variable(f"y_{e}_{w}"), w, e)])
    
    return z_e_u_e, z_e_w_e, y_e_e


def parallel_edge_update(rho):
    """
    Solve edge updates "in parallel". Except it's so trivial it's probably not
    worth parallelizing.
    
    Args:
        rho: scalar penalty parameter.
        
    Returns:
        np.ndarray: updated z variable set.
        float: elapsed solve time.
    """
    global z_global
    
    t_start = time.time()
    for e in E:
        z_e_u_e, z_e_w_e, y_e_e = edge_update(rho, e)
        z_global[consensus_manager.get_z_var_indices(Variable(f"z_{e}_{e[0]}_e"), e, e[0])] = z_e_u_e
        z_global[consensus_manager.get_z_var_indices(Variable(f"z_{e}_{e[1]}_e"), e, e[1])] = z_e_w_e
        z_global[consensus_manager.get_z_var_indices(Variable(f"y_{e}_e"), e)] = y_e_e
    t_elapsed = time.time() - t_start
    
    return t_elapsed
        

def dual_update():
    """
    Perform dual update ("mu-update") step.
    """
    return mu_global + (A @ x_global + B @ z_global - c)
    
    
def evaluate_primal_residual():
    return np.linalg.norm(A @ x_global + B @ z_global - c)


def evaluate_dual_residual(z_global_prev):
    return rho * np.linalg.norm(A.T @ B @ (z_global - z_global_prev))


def eps_pri(eps_abs, eps_rel, ord=2):
    return np.sqrt(x_global.shape[0]) * eps_abs + eps_rel * max(
        np.linalg.norm(A @ x_global, ord=ord),
        np.linalg.norm(B @ z_global, ord=ord),
        np.linalg.norm(c, ord=ord)
    )

    
def eps_dual(eps_abs, eps_rel, ord=2):
    return np.sqrt(mu_global.shape[0]) * eps_abs + eps_rel * np.linalg.norm(mu_global, ord=ord)


################################################################################
##### Main ADMM loop
################################################################################

rho = 1

# x vars
x_v_seq = [x_global[consensus_manager.get_x_v_var_indices_in_x()]]
z_v_seq = [x_global[consensus_manager.get_z_v_var_indices_in_x()]]
y_v_seq = [x_global[consensus_manager.get_y_v_var_indices_in_x()]]
z_e_u_v_seq = [x_global[consensus_manager.get_z_e_u_v_var_indices_in_x()]]
y_e_v_seq = [x_global[consensus_manager.get_y_e_v_var_indices_in_x()]]
# z vars  
z_e_u_e_seq = [z_global[consensus_manager.get_z_e_u_e_var_indices_in_z()]]
y_e_e_seq = [z_global[consensus_manager.get_y_e_e_var_indices_in_z()]]
# dual vars
mu_seq = [mu_global]

prev_z_global = z_global.copy()

rho_seq = [rho]
pri_res_seq = [evaluate_primal_residual()]
dual_res_seq = [evaluate_dual_residual(z_global)]

tau_incr = 2
tau_decr = 2
nu = 10
frac = 0.01  # after frac of iterations, stop updating rho

opt = False
eps_abs = 1e-4
eps_rel = 1e-3

it = 1
MAX_IT = 300

cumulative_solve_time = 0

while it <= MAX_IT:
    ##############################
    ### Vertex Updates
    ##############################
    x_global, vertex_solve_time = parallel_vertex_update(rho)
    cumulative_solve_time += vertex_solve_time

    if not np.all(np.isfinite(x_global)):
        print("BREAKING FOR Divergence")
        break
        
    # Update x history
    x_v_seq.append(x_global[consensus_manager.get_x_v_var_indices_in_x()])
    z_v_seq.append(x_global[consensus_manager.get_z_v_var_indices_in_x()])
    y_v_seq.append(x_global[consensus_manager.get_y_v_var_indices_in_x()])
    z_e_u_v_seq.append(x_global[consensus_manager.get_z_e_u_v_var_indices_in_x()])
    y_e_v_seq.append(x_global[consensus_manager.get_y_e_v_var_indices_in_x()])
    
    ##############################
    ### Edge Updates
    ##############################
    edge_solve_time = parallel_edge_update(rho)
    cumulative_solve_time += edge_solve_time

    if not np.all(np.isfinite(z_global)):
        print("BREAKING FOR Divergence")
        break
        
    # Update z history
    z_e_u_e_seq.append(z_global[consensus_manager.get_z_e_u_e_var_indices_in_z()])
    y_e_e_seq.append(z_global[consensus_manager.get_y_e_e_var_indices_in_z()])
    
    ##############################
    ### Dual Update
    ##############################
    mu_global = dual_update()
    
    # Update mu history
    mu_seq.append(mu_global)
    
    
    
    # Compute primal and dual residuals
    pri_res_seq.append(evaluate_primal_residual())
    dual_res_seq.append(evaluate_dual_residual(prev_z_global))
    prev_z_global = z_global.copy()
    
    # Update rho
    if  pri_res_seq[-1] >= nu * dual_res_seq[-1] and it < frac*MAX_IT:
        rho *= tau_incr
        mu_global /= tau_incr
    elif dual_res_seq[-1] >= nu* pri_res_seq[-1] and it < frac*MAX_IT:
        rho *= (1/tau_decr)
        mu_global *= tau_incr
    rho_seq.append(rho)
    
    # Check for convergence
    if pri_res_seq[-1] < eps_pri(eps_abs, eps_rel) and dual_res_seq[-1] < eps_dual(eps_abs, eps_rel):
        opt = True
    
    # Debug
    if it % 100 == 0 or it == MAX_IT or opt:
    # if it == MAX_IT:
        print(f"it = {it}/{MAX_IT}, {pri_res_seq[-1]=}, {dual_res_seq[-1]=}")
        fig, ax = plt.subplots(3)
        ax[0].loglog(rho_seq)
        ax[0].set_title("rho")
        ax[1].loglog(pri_res_seq)
        ax[1].set_title("pri_res")
        ax[2].loglog(dual_res_seq)
        ax[2].set_title("dual_res")
        plt.show()
        
    if opt:
        print("BREAKING FOR OPT")
        break
    
    it += 1
    
x_v_seq = np.array(x_v_seq)
z_v_seq = np.array(z_v_seq)
y_v_seq = np.array(y_v_seq)
z_e_u_v_seq = np.array(z_e_u_v_seq)
y_e_v_seq = np.array(y_e_v_seq)
z_e_u_e_seq = np.array(z_e_u_e_seq)
y_e_e_seq = np.array(y_e_e_seq)
mu_seq = np.array(mu_seq)

# Put most recent variables into dictionaries for rounding and visualization
x_v_sol = {v: x_v_seq[-1][2*i*n : 2*(i+1)*n] for i, v in enumerate(V)}
y_v_sol = {v: y_v_seq[-1][i] for i, v in enumerate(V)}
y_e_v_sol = {e: y_e_v_seq[-1][i] for i, e in enumerate(E)}

print(f"x_v: {x_v_sol}")
print(f"y_v: {y_v_sol}")
print(f"y_e: {y_e_v_sol}")

print(f"Total solve time: {cumulative_solve_time} s.")

visualize_results(As, bs, x_v_sol, y_v_sol)

rho_seq = np.array(rho_seq)
pri_res_seq = np.array(pri_res_seq)
dual_res_seq = np.array(dual_res_seq)