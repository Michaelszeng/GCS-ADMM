from pydrake.all import (
    MathematicalProgram, 
    Solve, 
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def convert_pt_to_polytope(pt, eps=1e-6):
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
    b = np.hstack([pt + eps, -pt + eps])
    
    return A, b


def build_graph(As, bs):
    """
    Generate vertex and edge sets and incidence lists for each vertex for the 
    GCS based on convex sets defined by As and bs.

    Args:
        As (np.ndarray): 3D numpy array where As[i] is the matrix A for polytope i.
        bs (np.ndarray): 2D numpy array where bs[i] is the vector b for polytope i.

    Returns:
        set: A set of vertices.
        set: A set of edges, where each edge is represented as a tuple of two vertices.
        dict: where I_v_in[v] is a list of edges incident to vertex v.
        dict: where I_v_out[v] is a list of edges incident to vertex v.
    """
    vertices = list(As.keys())  # vertex set
    edges = list()

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

    # Build edge set
    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                if check_overlap(As[v1], bs[v1], As[v2], bs[v2]):
                    edges.append((v1, v2))
    
    # Build incidence lists
    I_v_in = {v: [] for v in vertices}
    I_v_out = {v: [] for v in vertices}
    for e in edges:
        v, w = e
        I_v_out[v].append(e)
        I_v_in[w].append(e)

    return vertices, edges, I_v_in, I_v_out


def delta(v1, v2):
    """
    Delta variable in the GCS MICP formulation.
    
    Args:
        v1 (str or int): Start vertex key in adjacency list.
        v2 (str or int): End vertex key in adjacency list.
        
    Returns:
        Value of delta_{v1, v2} variable.
    """
    if v1 == v2 == 's' or v1 == v2 == 't':
        return 1
    return 0


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def visualize_results(As, bs, x_v, y_v):
    """
    Visualize 2D result of GCS piecewise-linear traj opt.

    Args:
        As: `As` dictionary from the test case definition.
        bs: `bs` dictionary from the test case definition.
        x_v: value of `x_v` dictionary from GCS MICP solution.
        y_v: value of `y_v` dictionary from GCS MICP solution.
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define colors for polytopes
    colors = plt.cm.tab10(np.linspace(0, 1, len(As)))
    
    # Initialize bounds for axis adjustment
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    for idx, (key, A) in enumerate(As.items()):
        b = bs[key]
        
        # Generate vertices of the polytope
        vertices = []
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[0]):
                # Solve for intersection points
                try:
                    point = np.linalg.solve(A[[i, j]], b[[i, j]])
                    if np.all(A @ point <= b + 1e-6):  # Check if the point is inside the polytope
                        vertices.append(point)
                except np.linalg.LinAlgError:
                    continue

        # Sort vertices in counterclockwise order
        if vertices:
            vertices = np.array(vertices)
            mean = np.mean(vertices, axis=0)
            angles = np.arctan2(vertices[:, 1] - mean[1], vertices[:, 0] - mean[0])
            vertices = vertices[np.argsort(angles)]

            # Add polytope as a polygon with a label for the legend
            if key != 's' and key != 't':
                polygon = Polygon(vertices, closed=True, alpha=0.4, color=colors[idx], label=f'Polytope {key}')
                ax.add_patch(polygon)

            # Update bounds
            x_min = min(x_min, vertices[:, 0].min())
            x_max = max(x_max, vertices[:, 0].max())
            y_min = min(y_min, vertices[:, 1].min())
            y_max = max(y_max, vertices[:, 1].max())

        # Plot points and line segments based on y_v
        if key in x_v and key in y_v:
            # Plot points if y_v[key] > 0.5
            if y_v[key] > 0.5:
                points = x_v[key].reshape(2, 2)  # Reshape to two points
                
                # Plot points without labels to avoid multiple legend entries
                ax.plot(points[:, 0], points[:, 1], 'o', color=colors[idx])
                
                # Draw line segment connecting the two points
                ax.plot(points[:, 0], points[:, 1], '-', color=colors[idx])

                # Update bounds
                x_min = min(x_min, points[:, 0].min())
                x_max = max(x_max, points[:, 0].max())
                y_min = min(y_min, points[:, 1].min())
                y_max = max(y_max, points[:, 1].max())

    # Configure plot limits to ensure visibility of all objects
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', adjustable='datalim')
    
    # Create a unique legend based on the polygon labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()
