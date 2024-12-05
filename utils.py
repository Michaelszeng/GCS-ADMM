from pydrake.all import (
    MathematicalProgram, 
    Solve, 
)

import numpy as np


def convert_pt_to_polytope(pt):
    """
    Converts a point into a degenerate polytope defined by Ax <= b.

    Args:
        pt (np.array): A 1D numpy array representing a point in n-dimensional space.

    Returns:
        A (np.ndarray): A 2n x n matrix defining the polytope constraints.
        b (np.ndarray): A 2n vector defining the polytope constraints.
    """
    n = len(pt)

    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.hstack([pt, -pt])
    
    return A, b


def build_graph(As, bs):
    """
    Generate vertex and edge sets for GCS based on convex sets defined by As and bs.

    Args:
        As (np.ndarray): 3D numpy array where As[i] is the matrix A for polytope i.
        bs (np.ndarray): 2D numpy array where bs[i] is the vector b for polytope i.

    Returns:
        set: A set of vertices.
        set: A set of edges, where each edge is represented as a tuple of two vertices.
    """
    vertices = set(As.keys())
    edges = set()

    def check_overlap(A1, b1, A2, b2):
        """
        Check if two polytopes defined by A1, b1 and A2, b2 overlap.
        """
        # Create combined polytope constraints
        A_combined = np.vstack([A1, A2])
        b_combined = np.hstack([b1, b2])
        
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(A1.shape[1], "x")
        
        for i in range(A_combined.shape[0]):
            prog.AddLinearConstraint(A_combined[i] @ x <= b_combined[i])
        
        # If there is a solution x in both polytopes, the polytopes overlap
        result = Solve(prog)
        return result.is_success()

    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                if check_overlap(As[v1], bs[v1], As[v2], bs[v2]):
                    edges.add((v1, v2))

    return vertices, edges


def delta(v1, v2):
    """
    Delta variable in the GCS MICP formulation.
    
    Args:
        v1 (str or int): Start vertex key in adjacency list.
        v2 (str or int): End vertex key in adjacency list.
        
    Returns:
        Value of delta_{v1, v2} variable.
    """
    if (v1 == 's' or v1 == 't') and v1 == v2:
        return 1
    return 0