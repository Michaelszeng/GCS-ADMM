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

# Public variable set to use to express consensus constraints and penalties
x_v_global = {}
z_v_global = {}
y_v_global = {}
x_v_e_global = {}
z_v_e_global = {}
y_e_global = {}

class ConsensusManager():
    """
    A class designed to take advantage of Drake's built-in optimization tools to
    more easily express the consensus constraints and penalties in standard
    Ax + Bz = c form.
    """
    def __init__(self, x_v, z_v, y_v, x_v_e, z_v_e, y_e):
        """
        Builds the MathematicalProgram containing all the variables in the split
        ADMM formulation.
        
        Args:
            Empty dictionaries for all variables in the split ADMM formulation.
        """    
        self.prog = MathematicalProgram()
        
        # Build variable set for x variables
        for v in V:
            x_v[v] = self.prog.NewContinuousVariables(2 * n, f'x_{v}')
            z_v[v] = self.prog.NewContinuousVariables(2 * n, f'z_{v}')
            y_v[v] = self.prog.NewBinaryVariables(1, f'y_{v}')[0]
        
        # Build variable set for z variables
        first_z_var = None
        for v in V:
            for e in I_v_in[v] + I_v_out[v]:
                if first_z_var is None:
                    first_z_var = x_v_e[(v, e)]
                x_v_e[(v, e)] = self.prog.NewContinuousVariables(2 * n, f'x_{v}_e_{e}')
                z_v_e[(v, e)] = self.prog.NewContinuousVariables(2 * n, f'z_{v}_e_{e}')
        for e in E:
            y_e[e] = self.prog.NewBinaryVariables(1, f'y_e_{e}')[0]
            
        # Make it easier to index into the z variables
        self.z_idx = self.prog.FindDecisionVariableIndex(first_z_var)
        
        self.x_v = x_v
        self.z_v = z_v
        self.y_v = y_v
        self.x_v_e = x_v_e
        self.z_v_e = z_v_e
        self.y_e = y_e
        
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
            # Vertex-Edge consensus constraints
            v, w = e
            # x_v = x_v_e
            consensus_costraints.append(self.x_v_e[(v, e)] == self.x_v[v])
            # x_w = x_w_e
            consensus_costraints.append(self.x_v_e[(w, e)] == self.x_v[w])
            
        for v in V:
            # Flow and Perspective Flow consensus constraints
            delta_sv = delta('s', v)
            delta_tv = delta('t', v)

            # y_v = sum_{e ∈ I_v_in} y_e + δ_{sv}
            consensus_costraints.append(self.y_v[v] == sum(self.y_e[e] for e in I_v_in[v]) + delta_sv)
            # y_v = sum_{e ∈ I_v_out} y_e + δ_{tv}
            consensus_costraints.append(self.y_v[v] == sum(self.y_e[e] for e in I_v_out[v]) + delta_tv)
            
            for d in range(2*n):   # 2n because z_v is 2n-dimensional
                # z_v = sum_in_z_v_e + δ_{sv} x_v
                consensus_costraints.append(self.z_v[v][d] == sum(self.z_v_e[(v, e)][d] for e in I_v_in[v]) + delta_sv * self.x_v[v][d])
                # z_v = sum_out_z_v_e + δ_{tv} x_v
                consensus_costraints.append(self.z_v[v][d] == sum(self.z_v_e[(v, e)][d] for e in I_v_out[v]) + delta_tv * self.x_v[v][d])
                
        # Now, construct A, B, c
        A = np.zeros((len(consensus_costraints), self.prog.num_vars()))
        B = np.zeros((len(consensus_costraints), self.z.num_vars()))
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
    
    def get_x_var_idx(self, var_name, v):
        """
        Helper funtion for each vertex update to find the index of the given
        variable in the x variable set.
        
        Args:
            var_name: str, name of the variable in the vertex update.
            v: vertex key for the variable/vertex update.
        """
        if "x_v" in var_name:
            var = x_v_global[v]
        elif "z_v" in var_name:
            var = z_v_global[v]
        elif "y_v" in var_name:
            var = y_v_global[v]
        else:
            raise ValueError("Invalid variable name.")
        
        idx = self.prog.FindDecisionVariableIndex(var)
        assert idx < self.z_idx  # Ensure the variable is in the x variable set
        return idx
    
    def get_z_var_idx(self, var_name, v, e):
        """
        Helper funtion for each edge update to find the index of the given
        variable in the z variable set.
        
        Args:
            var_name: str, name of the variable in the edge update.
            v: vertex key for the variable/edge update. (Note: not applicable for y_e)
            e: edge key for the variable/edge update.
        """
        if "x_v_e" in var_name:
            var = x_v_e_global[(v, e)]
        elif "z_v_e" in var_name:
            var = z_v_e_global[(v, e)]
        elif "y_e" in var_name:
            var = y_e_global[e]
        
        idx = self.prog.FindDecisionVariableIndex(var) - self.z_idx
        assert idx >= 0  # Ensure the variable is in the z variable set
        return idx
    
    def get_num_x_vars(self):
        return self.z_idx
    
    def get_num_z_vars(self):
        return self.prog.num_vars() - self.z_idx
    
    def get_num_mu_vars(self):
        if self.A is None:
            raise ValueError("Call build_A_B_c_consensus_matrices() first.")
        return self.A.shape[0]  # Number of rows of A = number of consensus constraints = number of mu variables


# Build consensus manager to handle construction of A, B, c matrices for consensus constraints and penalties
consensus_manager = ConsensusManager(x_v_global, z_v_global, y_v_global, x_v_e_global, z_v_e_global, y_e_global)
A, B, c = consensus_manager.build_A_B_c_consensus_matrices()  # Just build these once; they remain constant throughout optimization

# Variables to store current global values of split x and z variables
# Consists of x_v, z_v, y_v
x_global = np.zeros(consensus_manager.get_num_x_vars())
# Consists of x_v_e, z_v_e, y_e
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
    x_v = prog.NewContinuousVariables(2 * n, f'x_v')
    z_v = prog.NewContinuousVariables(2 * n, f'z_v')
    y_v = prog.NewContinuousVariables(1, f'y_v')[0]  # Relax y_v to 0 <= y_v <= 1
    prog.AddBoundingBoxConstraint(0, 1, y_v)
    
    # Path Length Penalty: ||z_v1 - z_v2||^2
    z_v1 = z_v[:n]
    z_v2 = z_v[n:]
    A = np.hstack([np.eye(z_v1.shape[0]), -np.eye(z_v2.shape[0])])
    b = np.zeros(A.shape[0])
    prog.AddL2NormCost(A, b, np.hstack([z_v1, z_v2]))
    
    # Concensus Constraint Penalty: (rho/2) * ||Ax + Bz + mu||^2
    # Define x vector that contain fixed values for all but the variables corresponding to v
    x = x_global.copy()
    x[consensus_manager.get_x_var_idx(x_v, v)] = x_v
    x[consensus_manager.get_x_var_idx(z_v, v)] = z_v
    x[consensus_manager.get_x_var_idx(y_v, v)] = y_v
    # z and mu are all fixed
    prog.AddCost((rho/2) * (A @ x + B @ z_global - c + mu_global).T @ (A @ x + B @ z_global - c + mu_global))
    
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
            
    result = Solve(prog)
    
    if result.is_success():
        # Solution retreival
        x_v_sol = result.GetSolution(x_v)
        z_v_sol = result.GetSolution(z_v)
        y_v_sol = result.GetSolution(y_v)

        print(f"x_v_sol: NEW: {x_v_sol}. OLD: {x_global[consensus_manager.get_x_var_idx(x_v, v)]}.\n")
        print(f"z_v_sol: NEW: {z_v_sol}. OLD: {x_global[consensus_manager.get_x_var_idx(z_v, v)]}.\n")
        print(f"y_v_sol: NEW: {y_v_sol}. OLD: {x_global[consensus_manager.get_x_var_idx(y_v, v)]}.\n")
        
        # Update global values
        x_global[consensus_manager.get_x_var_idx(x_v, v)] = x_v_sol
        x_global[consensus_manager.get_x_var_idx(z_v, v)] = z_v_sol
        x_global[consensus_manager.get_x_var_idx(y_v, v)] = y_v_sol
        
    else:
        print("solve failed.")
        print(f"{result.get_solution_result()}")
        print(f"{result.GetInfeasibleConstraintNames(prog)}")
        for constraint_binding in result.GetInfeasibleConstraints(prog):
            print(f"{constraint_binding.variables()}")


def edge_update(rho, e):
    """
    Perform edge update ("z-update") step for a single vertex edge e = (v,w).
    
    Args:
        rho: scalar penalty parameter.
        e: edge key for the edge being updated.
    """
    prog = MathematicalProgram()
    x_v_e = prog.NewContinuousVariables(2 * n, f'x_v_e')
    z_v_e = prog.NewContinuousVariables(2 * n, f'z_v_e')
    x_w_e = prog.NewContinuousVariables(2 * n, f'x_w_e')
    z_w_e = prog.NewContinuousVariables(2 * n, f'z_w_e')
    y_e = prog.NewContinuousVariables(1, f'y_e')[0]  # Relax y_e to 0 <= y_e <= 1
    prog.AddBoundingBoxConstraint(0, 1, y_e)
    
    # Edge Activation Penalty: 1e-4 * y_e
    prog.AddCost(1e-4 * y_e)
    
    # Concensus Constraint Penalty: (rho/2) * ||Ax + Bz + mu||^2
    # Define z vector that contain fixed values for all but the variables corresponding to e
    z = z_global.copy()
    z[consensus_manager.get_z_var_idx(x_v_e, e[0], e)] = x_v_e
    z[consensus_manager.get_z_var_idx(z_v_e, e[0], e)] = z_v_e
    z[consensus_manager.get_z_var_idx(x_w_e, e[1], e)] = x_w_e
    z[consensus_manager.get_z_var_idx(z_w_e, e[1], e)] = z_w_e
    z[consensus_manager.get_z_var_idx(y_e, None, e)] = y_e
    # x and mu are all fixed
    prog.AddCost((rho/2) * (A @ x_global + B @ z - c + mu_global).T @ (A @ x_global + B @ z - c + mu_global))
    
    # Point Containment Constraints (for both points corresponding to e)
    for v in e:  # e = (v,w)
        m = As[v].shape[0]
        
        # Select whether to constraint x_v_e and z_v_e or x_w_e and z_w_e
        if v == e[0]:
            z_v_e_active = z_v_e
            x_v_e_active = x_v_e
        else:
            z_v_e_active = z_w_e
            x_v_e_active = x_w_e
                
        for i in range(2):
            idx = slice(i * n, (i + 1) * n)
            
            # Constraint 1: A_v z^e_{v,i} ≤ y_e b_v
            for j in range(m):
                prog.AddConstraint(As[v][j] @ z_v_e_active[idx] <= y_e * bs[v][j])
                
            # Constraint 2: A_v (x^e_{v,i} - z^e_{v,i}) ≤ (1 - y_e) b_v
            for j in range(m):
                prog.AddConstraint(As[v][j] @ (x_v_e_active[idx] - z_v_e_active[idx]) <= (1 - y_e) * bs[v][j])
            
    # Path Continuity Constraint: z^e_{v,2} = z^e_{w,1}
    for d in range(n):
        prog.AddConstraint(z_v_e[n+d] == z_w_e[d])
            
    result = Solve(prog)
    
    if result.is_success():
        # Solution retreival
        x_v_e_sol = result.GetSolution(x_v_e)
        z_v_e_sol = result.GetSolution(z_v_e)
        x_w_e_sol = result.GetSolution(x_w_e)
        z_w_e_sol = result.GetSolution(z_w_e)
        y_e_sol = result.GetSolution(y_e)

        print(f"x_v_e_sol: NEW: {x_v_e_sol}. OLD: {x_global[consensus_manager.get_z_var_idx(x_v_e, e[0], e)]}.\n")
        print(f"z_v_e_sol: NEW: {z_v_e_sol}. OLD: {x_global[consensus_manager.get_z_var_idx(z_v_e, e[0], e)]}.\n")
        print(f"x_v_e_sol: NEW: {x_w_e_sol}. OLD: {x_global[consensus_manager.get_z_var_idx(x_w_e, e[1], e)]}.\n")
        print(f"z_v_e_sol: NEW: {z_w_e_sol}. OLD: {x_global[consensus_manager.get_z_var_idx(z_w_e, e[1], e)]}.\n")
        print(f"y_e_sol:   NEW: {y_e_sol}. OLD: {x_global[consensus_manager.get_z_var_idx(y_e, None, e)]}.\n")
        
        # Update global values
        z_global[consensus_manager.get_z_var_idx(x_v_e, e[0], e)] = x_v_e_sol
        z_global[consensus_manager.get_z_var_idx(z_v_e, e[0], e)] = z_v_e_sol
        z_global[consensus_manager.get_z_var_idx(x_w_e, e[1], e)] = x_w_e_sol
        z_global[consensus_manager.get_z_var_idx(z_w_e, e[1], e)] = z_w_e_sol
        z_global[consensus_manager.get_z_var_idx(y_e, None, e)] = y_e_sol
        
    else:
        print("solve failed.")
        print(f"{result.get_solution_result()}")
        print(f"{result.GetInfeasibleConstraintNames(prog)}")
        for constraint_binding in result.GetInfeasibleConstraints(prog):
            print(f"{constraint_binding.variables()}")


def dual_update():
    """
    Perform dual update ("mu-update") step.
    """
    mu_global = mu_global + (A @ x_global + B @ z_global - c)
    
    
def evaluate_primal_residual():
    return np.linalg.norm(A @ x_global + B @ z_global - c)


def evaluate_dual_residual(z_global_prev):
    return rho * np.linalg.norm(A.T @ B @ (z_global - z_global_prev))


################################################################################
##### Main ADMM loop
################################################################################

rho = 1

x_v_seq = [x_global[:2*n]]
z_v_seq = [x_global[2*n:4*n]]
y_v_seq = [x_global[4*n:]]
x_v_e_seq = [z_global[:2*n]]
z_v_e_seq = [z_global[2*n:4*n]]
y_e_seq = [z_global[4*n:]]
mu_seq = [mu_global]

rho_seq = [rho]
pri_res_seq = [evaluate_primal_residual()]
dual_res_seq = [evaluate_dual_residual(z_global)]

prev_z_global = z_global.copy()

tau_incr = 2
tau_decr = 2
nu = 10
frac = 0.01  # after frac of iterations, stop updating rho
it = 0
MAX_IT = 100

while it < MAX_IT:
    ##############################
    ### Vertex Updates
    ##############################
    for v in V:
        vertex_update(rho, v)
        
    if not np.all(np.isfinite(x_global)):
        print("BREAKING FOR Divergence")
        break
        
    # Update x history
    x_v_seq.append(x_global[:2*n])
    z_v_seq.append(x_global[2*n:4*n])
    y_v_seq.append(x_global[4*n:])
    
    ##############################
    ### Edge Updates
    ##############################
    for e in E:
        edge_update(rho, e)
        
    if not np.all(np.isfinite(z_global)):
        print("BREAKING FOR Divergence")
        break
        
    # Update z history
    x_v_e_seq.append(z_global[:2*n])
    z_v_e_seq.append(z_global[2*n:4*n])
    y_e_seq.append(z_global[4*n:])
    
    ##############################
    ### Dual Update
    ##############################
    dual_update()
    
    # Update mu history
    mu_seq.append(mu_global)
    
    
    
    # Compute primal and dual residuals
    pri_res_seq.append(evaluate_primal_residual())
    dual_res_seq.append(evaluate_dual_residual(prev_z_global))
    prev_z_global = z_global.copy()
    
    # Update rho
    if  pri_res_seq[-1] >= nu * dual_res_seq[-1] and it < frac*MAX_IT:
        rho *= tau_incr
    elif dual_res_seq[-1] >= nu* pri_res_seq[-1] and it < frac*MAX_IT:
        rho *= (1/tau_decr)
    rho_seq.append(rho)
    
    # Debug
    if it % 10 == 0:
        print(f"it = {it+1}/{MAX_IT}, {pri_res_seq[-1]=}, {dual_res_seq[-1]=}")
        fig, ax = plt.subplots(3)
        ax[0].loglog(rho_seq)
        ax[0].set_title("rho")
        ax[1].loglog(pri_res_seq)
        ax[1].set_title("pri_res")
        ax[2].loglog(pri_res_seq)
        ax[2].set_title("dual_res")
        plt.show()
    
    it += 1
    
x_v_seq = np.array(x_v_seq)
z_v_seq = np.array(z_v_seq)
y_v_seq = np.array(y_v_seq)
x_v_e_seq = np.array(x_v_e_seq)
z_v_e_seq = np.array(z_v_e_seq)
y_e_seq = np.array(y_e_seq)
mu_seq = np.array(mu_seq)

rho_seq = np.array(rho_seq)
pri_res_seq = np.array(pri_res_seq)
dual_res_seq = np.array(dual_res_seq)