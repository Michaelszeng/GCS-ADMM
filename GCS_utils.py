from pydrake.all import (
    MathematicalProgram, 
    Solve, 
)

import numpy as np
import sys
import os
import time

from utils import *

current_folder = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_folder, "test_data")
sys.path.append(test_data_path)

def solve_convex_restriction(As, bs, n, V, E, y_v, y_e):
    """
    Solve the GCS problem given the path. This is a simple convex program. 
    """
    prog = MathematicalProgram()

    ################################################################################
    ##### Variable Definitions
    ################################################################################
    x_v = {}
    
    # Variables for each vertex v ∈ V
    for v in V:
        x_v[v] = prog.NewContinuousVariables(2 * n, f'x_{v}')
                
    ################################################################################
    ##### Cost
    ################################################################################
    # Path length penalty: sum_{v ∈ V} ||y_v x_v1 - y_v x_v2||^2
    for v in V:
        z_v1 = x_v[v][:n]
        z_v2 = x_v[v][n:]
        A = y_v[v] * np.hstack([np.eye(z_v1.shape[0]), -np.eye(z_v2.shape[0])])
        b = np.zeros(A.shape[0])
        prog.AddL2NormCost(A, b, np.hstack([z_v1, z_v2]))        

    ################################################################################
    ##### Constraints
    ################################################################################
    # Vertex Point Containment Constraints
    for v in V:
        m = As[v].shape[0]
        for i in range(2):
            idx = slice(i * n, (i + 1) * n)

            # y_v[v] * A_v x_{v,i} ≤ y_v * b_v
            for j in range(m):
                prog.AddConstraint(y_v[v] * As[v][j] @ x_v[v][idx] <= y_v[v] * bs[v][j])
                
    # Path Continuity Constraints
    for e in E:
        v, w = e
        # Constraint 5: y_e * x_{v,2}^e = y_e * x_{w,1}^e for each edge e = (v, w)
        for d in range(n):  # n because we only check equivalence of one point in each z_v_e (which represents two points)
            prog.AddConstraint(y_e[e] * x_v[v][n+d] == y_e[e] * x_v[w][d])
        
    ################################################################################
    ##### Solve
    ################################################################################
    print("Beginning Convex Restriction Solve.")
    start = time.time()
    result = Solve(prog)
    print(f"Solve Time: {time.time() - start}")
    print(f"Solved using: {result.get_solver_id().name()}")

    if result.is_success():
        # Solution retreival
        x_v_sol = {}
        for v in V:
            x_v_sol[v] = result.GetSolution(x_v[v])
            
        cost = result.get_optimal_cost()
        
        return cost, x_v_sol, y_v
        
    else:
        print("Convex restriction solve failed.")
        print(f"{result.get_solution_result()}")
        print(f"{result.GetInfeasibleConstraintNames(prog)}")
        for constraint_binding in result.GetInfeasibleConstraints(prog):
            print(f"{constraint_binding.variables()}")
            
        return float('inf'), None, None


def rounding(y_e_sol, V, E, I_v_out, As, bs, n, N=5, M=20, solve_convex_restriction=solve_convex_restriction):
    """
    Perform rounding steps using GCS convex relaxation result to obtain feasible
    solution (i.e. binary y variables).
    
    Args:
        y_e_sol : dict result from GCS; mapping edges (u, v) to probabilities in [0, 1].
        I_v_out : dict result from `build_grph`; mapping each vertex v to a list of edges (v, w) representing its outgoing edges.
        N : int, max number of distinct paths to collect before stopping (default: 10).
        M : int, max number of random trials (default: 100).
        solve_convex_restriction : callable; solves the convex restriction given a fixed path and returns the cost or `float('inf')` if infeasible.
        
    Returns
        best_path : list or None; best path found as a list of vertices, or None if no path found.
        best_cost : float; cost of the best path found or inf if none found.
    """
    
    def find_path_via_random_dfs():
        path = ['s']
        visited = {'s'}
        
        def dfs(current):
            if current == 't':
                return True
            
            # Collect feasible outgoing edges and their probabilities
            edges = [(current, w) for (current, w) in I_v_out.get(current, [])
                     if w not in visited and y_e_sol.get((current, w), 0) > 1e-15]
            
            if not edges:
                return False
            
            probs = np.array([y_e_sol[e] for e in edges], dtype=float)
            total = probs.sum()
            if total < 1e-15:
                return False
            
            # Normalize and sample one edge according to probabilities
            probs /= total
            r = np.random.rand()
            idx = np.searchsorted(np.cumsum(probs), r)
            
            chosen_edge = edges[idx]
            visited.add(chosen_edge[1])
            path.append(chosen_edge[1])
            
            if dfs(chosen_edge[1]):
                return True
            
            # Backtrack
            visited.remove(chosen_edge[1])
            path.pop()
            return False
        
        return path if dfs('s') else None

    distinct_paths = set()
    candidate_solutions = []
    
    for _ in range(M):
        # Stop if we have enough candidate solutions
        if len(candidate_solutions) >= N:
            break
        p = find_path_via_random_dfs()
        if p is not None:
            p_tuple = tuple(p)
            if p_tuple not in distinct_paths:
                distinct_paths.add(p_tuple)
                
                # Construct y_v and y_e from path p
                y_v = {v: 0 for v in V}
                for v in p:
                    y_v[v] = 1

                y_e = {e: 0 for e in E}
                # Mark edges that form the path
                for i in range(len(p)-1):
                    edge = (p[i], p[i+1])
                    y_e[edge] = 1
                
                # Solve convex restriction
                cost, x_v_sol, y_v_sol = solve_convex_restriction(As, bs, n, V, E, y_v, y_e)
                if cost != float('inf'):
                    candidate_solutions.append((cost, x_v_sol, y_v_sol))
    
    if not candidate_solutions:
        print("Rounding failed to find any feasible paths.")
        return float('inf'), None, None
    best_cost, best_x_v_sol, best_y_v_sol = min(candidate_solutions, key=lambda x: x[0])
    return best_cost, best_x_v_sol, best_y_v_sol


def compute_cost(z_v_sol, y_e_sol):
    """
    Computes the cost of the optimization problem given the values of z_v and y_e variables.

    Parameters:
    - z_v_sol (dict): Dictionary mapping each vertex v to its z_v values (2*n-dimensional numpy array).
    - y_e_sol (dict): Dictionary mapping each edge e to its y_e value (float).

    Returns:
    - float: The computed cost based on the given variable values.
    """
    path_length_penalty = 0.0
    edge_activation_penalty = 0.0

    # Compute path length penalty: sum_{v ∈ V} ||z_v1 - z_v2||
    for v, z_v in z_v_sol.items():
        n = z_v.shape[0] // 2
        z_v1 = z_v[:n]
        z_v2 = z_v[n:]
        path_length_penalty += np.linalg.norm(z_v1 - z_v2)

    # Compute edge activation penalty: sum_{e ∈ E} 1e-4 * y_e
    for e, y_e in y_e_sol.items():
        edge_activation_penalty += 1e-4 * y_e

    # Total cost
    total_cost = path_length_penalty + edge_activation_penalty
    return total_cost