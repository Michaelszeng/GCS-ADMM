from pydrake.all import (
    MathematicalProgram, 
    Solve, 
    Expression,
    Evaluate,
    Constraint,
    Variable,
    eq,
    le,
    ge,
)
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
# Initialize variables to 0
x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y = 4, 4, -2, 4, 4, 4, 4, -2, -2, 0  # feasible initial guess (so indicator costs don't explode)

# # Symbolic Expression for Lagrangian
# x_1_sym = np.array([Variable("x_1")])
# x_2_sym = np.array([Variable("x_2")])
# x_3_sym = np.array([Variable("x_3")])
# z1_1_sym = np.array([Variable("z1_1")])
# z1_2_sym = np.array([Variable("z1_2")])
# z2_2_sym = np.array([Variable("z2_2")])
# z2_3_sym = np.array([Variable("z2_3")])
# z3_1_sym = np.array([Variable("z3_1")])
# z3_3_sym = np.array([Variable("z3_3")])
# y_sym = np.array([Variable("y")])

def indicator_cost(f, x):
    """
    Return penalty if f is violated; 0 otherwise.

    f is the constraint function.
    x is a list of all variables fed to the constraint function.
    """
    penalty = 1e6
    

def augmented_lagrangian(x_1_sym, x_2_sym, x_3_sym, z1_1_sym, z3_1_sym, z1_2_sym, z2_2_sym, z2_3_sym,  z3_3_sym, y):
    x_sym = np.vstack((x_1_sym, x_2_sym, x_3_sym))
    z_sym = np.vstack((z1_1_sym, z3_1_sym, z1_2_sym, z2_2_sym, z2_3_sym, z3_3_sym))
    
    return x1_cost(x_1_sym) + x2_cost(x_2_sym) + x3_cost(x_3_sym) + x1_x2_cost(z1_1_sym, z1_2_sym) + x2_x3_cost(z2_2_sym, z2_3_sym) + x1_x3_cost(z3_1_sym, z3_3_sym) \
        + indicator_cost(x1_constraint, [x_1_sym]) + indicator_cost(x2_constraint, [x_2_sym]) + indicator_cost(x3_constraint, [x_3_sym]) \
        + indicator_cost(x1_x2_constraint, [x_1_sym, x_2_sym]) + indicator_cost(x2_x3_constraint, [x_2_sym, x_3_sym]) + indicator_cost(x1_x3_constraint, [x_1_sym, x_3_sym]) \
        + np.dot(y, (A @ x_sym + B @ z_sym - c)) \
        + rho/2 * np.dot((A @ x_sym + B @ z_sym - c), (A @ x_sym + B @ z_sym - c))
    

def argmin_x1(x_2, x_3, z1_1, z3_1, z1_2, z2_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    x_1 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"x1 update: Is solved successfully: {result.is_success()}")
    print(f"x1 update: optimal cost: {result.get_optimal_cost()}")
    print(f"x1 update: x1 optimal value: {result.GetSolution(x_1)}")
    return result.GetSolution(x_1)

def argmin_x2(x_1, x_3, z1_1, z3_1, z1_2, z2_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    x_2 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"x2 update: Is solved successfully: {result.is_success()}")
    print(f"x2 update: optimal cost: {result.get_optimal_cost()}")
    print(f"x2 update: x2 optimal value: {result.GetSolution(x_2)}")
    return result.GetSolution(x_2)

def argmin_x3(x_1, x_2, z1_1, z3_1, z1_2, z2_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    x_3 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"x3 update: Is solved successfully: {result.is_success()}")
    print(f"x3 update: optimal cost: {result.get_optimal_cost()}")
    print(f"x3 update: x3 optimal value: {result.GetSolution(x_3)}")
    return result.GetSolution(x_3)

def argmin_z1_1(x_1, x_2, x_3, z3_1, z1_2, z2_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    z1_1 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"z1_1 update: Is solved successfully: {result.is_success()}")
    print(f"z1_1 update: optimal cost: {result.get_optimal_cost()}")
    print(f"z1_1 update: z1_1 optimal value: {result.GetSolution(z1_1)}")
    return result.GetSolution(z1_1)

def argmin_z3_1(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    z3_1 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"z3_1 update: Is solved successfully: {result.is_success()}")
    print(f"z3_1 update: optimal cost: {result.get_optimal_cost()}")
    print(f"z3_1 update: z3_1 optimal value: {result.GetSolution(z3_1)}")
    return result.GetSolution(z3_1)

def argmin_z1_2(x_1, x_2, x_3, z1_1, z3_1, z2_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    z1_2 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"z1_2 update: Is solved successfully: {result.is_success()}")
    print(f"z1_2 update: optimal cost: {result.get_optimal_cost()}")
    print(f"z1_2 update: z1_2 optimal value: {result.GetSolution(z1_2)}")
    return result.GetSolution(z1_2)

def argmin_z2_2(x_1, x_2, x_3, z1_1, z3_1, z1_2, z2_3, z3_3, y):
    sub_prog = MathematicalProgram()
    z2_2 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"z2_2 update: Is solved successfully: {result.is_success()}")
    print(f"z2_2 update: optimal cost: {result.get_optimal_cost()}")
    print(f"z2_2 update: z2_2 optimal value: {result.GetSolution(z2_2)}")
    return result.GetSolution(z2_2)

def argmin_z2_3(x_1, x_2, x_3, z1_1, z3_1, z1_2, z2_2, z3_3, y):
    sub_prog = MathematicalProgram()
    z2_3 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"z2_3 update: Is solved successfully: {result.is_success()}")
    print(f"z2_3 update: optimal cost: {result.get_optimal_cost()}")
    print(f"z2_3 update: z2_3 optimal value: {result.GetSolution(z2_3)}")
    return result.GetSolution(z2_3)

def argmin_z3_3(x_1, x_2, x_3, z1_1, z3_1, z1_2, z2_2, z2_3, y):
    sub_prog = MathematicalProgram()
    z3_3 = sub_prog.NewContinuousVariables(n)
    sub_prog.AddCost(augmented_lagrangian(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y))
    result = Solve(prog)
    print(f"z3_3 update: Is solved successfully: {result.is_success()}")
    print(f"z3_3 update: optimal cost: {result.get_optimal_cost()}")
    print(f"z3_3 update: z3_3 optimal value: {result.GetSolution(z3_3)}")
    return result.GetSolution(z3_3)

A = np.array([[1,0,0],
                  [1,0,0],
                  [0,1,0],
                  [0,1,0],
                  [0,0,1],
                  [0,0,1]])
B = -np.eye(6)
c = 0

rho=1
iter_limit = 100
for i in range(iter_limit):
    x_1 = argmin_x1(x_2, x_3, z1_1, z3_1, z1_2, z2_2, z2_3, z3_3, y)
    
    x_2 = argmin_x2(x_1, x_3, z1_1, z3_1, z1_2, z2_2, z2_3, z3_3, y)
    
    x_3 = argmin_x3(x_1, x_2, z1_1, z3_1, z1_2, z2_2, z2_3, z3_3, y)
    
    z1_1 = argmin_z1_1(x_1, x_2, x_3, z3_1, z1_2, z2_2, z2_3, z3_3, y)
    
    z3_1 = argmin_z3_1(x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_3, y)
    
    z1_2 = argmin_z1_2(x_1, x_2, x_3, z1_1, z3_1, z2_2, z2_3, z3_3, y)
    
    z2_2 = argmin_z2_2(x_1, x_2, x_3, z1_1, z3_1, z1_2, z2_3, z3_3, y)
    
    z2_3 = argmin_z2_3(x_1, x_2, x_3, z1_1, z3_1, z1_2, z2_2, z3_3, y)
    
    z3_3 = argmin_z3_3(x_1, x_2, x_3, z1_1, z3_1, z1_2, z2_2, z2_3, y)
    
    x = np.vstack((x_1, x_2, x_3))
    z = np.vstack((z1_1, z3_1, z1_2, z2_2, z2_3, z3_3))
    y = y + rho*(A @ x + B @ z - c)
    
    print(f"x_{i}: {x}")