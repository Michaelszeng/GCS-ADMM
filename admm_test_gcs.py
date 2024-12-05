from pydrake.all import (
    MathematicalProgram, 
    Solve, 
    Expression,
    Formula,
    Evaluate,
    Constraint,
    Variable,
    eq,
    le,
    ge,
    logical_and,
)
from pydrake.symbolic import max as sym_max, is_true

import numpy as np

################################################################################
# Problem Setup
################################################################################
n = 1  # number of decision variables per convex set

prog = MathematicalProgram()
x1 = prog.NewContinuousVariables(n)
x2 = prog.NewContinuousVariables(n)
x3 = prog.NewContinuousVariables(n)

def x1_cost(x):
    # Return an Expression
    c = [1]
    return np.dot(c, x)

def x2_cost(x):
    # Return an Expression
    c = [1]
    return np.dot(c, x)

def x3_cost(x):
    # Return an Expression
    c = [1]
    return np.dot(c, x)

def x1_x2_cost(x, x_):
    # Return an Expression
    c = [-1,-1]
    return np.dot(c, np.hstack((x, x_)))

def x1_x3_cost(x, x_):
    # Return an Expression
    c = [1,-1]
    return np.dot(c, np.hstack((x, x_)))

def x2_x3_cost(x, x_):
    # Return an Expression
    c = [1,-1]
    return np.dot(c, np.hstack((x, x_)))

def x1_constraint(x):
    A = np.array([[1],[-1]])
    b = np.array([10,0])
    return le(A @ x, b)

def x2_constraint(x):
    A = np.array([[1],[-1]])
    b = np.array([6,-4])
    return le(A @ x, b)

def x3_constraint(x):
    A = np.array([[1],[-1]])
    b = np.array([2,2])
    return le(A @ x, b)

def x1_x2_constraint(x, x_):
    A = np.array([[-1]])
    B = np.array([[1]])
    c = np.array([0])
    return le(A @ x + B @ x_, c)

def x2_x3_constraint(x, x_):
    A = np.array([[-1]])
    B = np.array([[-1]])
    c = np.array([-2])
    return le(A @ x + B @ x_, c)

def x1_x3_constraint(x, x_):
    A = np.array([[1]])
    B = np.array([[1]])
    c = np.array([3])
    return le(A @ x + B @ x_, c)


################################################################################
# Complete GCS solved using LP
################################################################################
prog.AddCost(x1_cost(x1))
prog.AddCost(x2_cost(x2))
prog.AddCost(x3_cost(x3))
prog.AddCost(x1_x2_cost(x1, x2))
prog.AddCost(x1_x3_cost(x1, x3))
prog.AddCost(x2_x3_cost(x2, x3))

prog.AddConstraint(x1_constraint(x1))
prog.AddConstraint(x2_constraint(x2))
prog.AddConstraint(x3_constraint(x3))
prog.AddConstraint(x1_x2_constraint(x1, x2))
prog.AddConstraint(x2_x3_constraint(x2, x3))
prog.AddConstraint(x1_x3_constraint(x1, x3))

result = Solve(prog)

print(f"Is solved successfully: {result.is_success()}")
print(f"x1 optimal value: {result.GetSolution(x1)}")
print(f"x2 optimal value: {result.GetSolution(x2)}")
print(f"x3 optimal value: {result.GetSolution(x3)}")
print(f"optimal cost: {result.get_optimal_cost()}")





################################################################################
# GCS solved with ADMM
################################################################################
# Initialize variables to feasible initial guesses
x_1, x_2, x_3 = np.array([4]), np.array([4]), np.array([-2])
z1_1, z1_2, z2_2, z2_3, z3_1, z3_3 = x_1, x_2, x_2, x_3, x_1, x_3
y = np.zeros((6,1))

# Constraint values (x=z)
A = np.array([[1,0,0],
                  [1,0,0],
                  [0,1,0],
                  [0,1,0],
                  [0,0,1],
                  [0,0,1]])
B = -np.eye(6)
c = np.zeros((6,1))

def indicator_cost(f, x):
    """
    Return penalty * sum of constraint violations.

    f is the constraint function.
    x is a list of all variables fed to the constraint function.
    """
    penalty = 1e6
    constraints = f(*x)
    if isinstance(constraints[0], Formula):
        constraints
    else:
        return all(constraints)
    return is_true(logical_and(*constraints))
    print(f"violations: {violations}")
    return 0

indicator_cost(x1_constraint, np.array([[1]]))
sub_prog = MathematicalProgram()
var = sub_prog.NewContinuousVariables(1, "x1")
indicator_cost(x1_constraint, [var])

def augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y):
    x = np.vstack((x_1, x_2, x_3))
    z = np.vstack((z1_1, z3_1, z1_2, z2_2, z2_3, z3_3))
    
    lagrangian = (x1_cost(x_1) + x2_cost(x_2) + x3_cost(x_3) + x1_x2_cost(z1_1, z1_2) + x2_x3_cost(z2_2, z2_3) + x1_x3_cost(z3_1, z3_3) +
                  indicator_cost(x1_constraint, [x_1]) + indicator_cost(x2_constraint, [x_2]) + indicator_cost(x3_constraint, [x_3]) +
                  indicator_cost(x1_x2_constraint, [x_1, x_2]) + indicator_cost(x2_x3_constraint, [x_2, x_3]) + indicator_cost(x1_x3_constraint, [x_1, x_3]) +
                  y.T @ (A @ x + B @ z - c) +
                  (rho/2) * ((A @ x + B @ z - c).T @ (A @ x + B @ z - c))
                 )
    return lagrangian[0][0]

def argmin_variable(var_name, vars_dict, y):
    sub_prog = MathematicalProgram()
    var = sub_prog.NewContinuousVariables(n, var_name)
    
    # Update the variable to be optimized
    vars_dict[var_name] = var

    sub_prog.AddCost(augmented_lagrangian(**vars_dict, y=y))
    result = Solve(sub_prog)
    if not result.is_success():
        print(f"Optimization for {var_name} failed.")
    return result.GetSolution(var)

rho = 1
iter_limit = 0
for i in range(iter_limit):
    fixed_vars = {
        'x_1': x_1, 'x_2': x_2, 'x_3': x_3, 
        'z1_1': z1_1, 'z1_2': z1_2, 'z2_2': z2_2, 
        'z2_3': z2_3, 'z3_1': z3_1, 'z3_3': z3_3
    }
    
    # Update x_1
    x_1 = argmin_variable('x_1', fixed_vars, y)
    fixed_vars['x_1'] = x_1
        
    # Update x_2
    x_2 = argmin_variable('x_2', fixed_vars, y)
    fixed_vars['x_2'] = x_2
    
    # Update x_3
    x_3 = argmin_variable('x_3', fixed_vars, y)
    fixed_vars['x_3'] = x_3
    
    # Update z1_1
    z1_1 = argmin_variable('z1_1', fixed_vars, y)
    fixed_vars['z1_1'] = z1_1
    
    # Update z1_2
    z1_2 = argmin_variable('z1_2', fixed_vars, y)
    fixed_vars['z1_2'] = z1_2
    
    # Update z2_2
    z2_2 = argmin_variable('z2_2', fixed_vars, y)
    fixed_vars['z2_2'] = z2_2
    
    # Update z2_3
    z2_3 = argmin_variable('z2_3', fixed_vars, y)
    fixed_vars['z2_3'] = z2_3
    
    # Update z3_1
    z3_1 = argmin_variable('z3_1', fixed_vars, y)
    fixed_vars['z3_1'] = z3_1
    
    # Update z3_3
    z3_3 = argmin_variable('z3_3', fixed_vars, y)
    fixed_vars['z3_3'] = z3_3
    
    # Update dual variable y
    x = np.vstack((x_1, x_2, x_3))
    z = np.vstack((z1_1, z3_1, z1_2, z2_2, z2_3, z3_3))
    y = y + rho*(A @ x + B @ z - c)
    
    print(f"x_{i}: {x.flatten()}")