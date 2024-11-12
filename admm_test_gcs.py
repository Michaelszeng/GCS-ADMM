from pydrake.all import (
    MathematicalProgram, 
    Solve, 
    BoundingBoxConstraint,
    Expression,
    Constraint,
    Variable,
    LinearConstraint,
    ExpressionConstraint,
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
    c = [1]*n
    return np.dot(c, x)

def x2_cost(x):
    # Return an Expression
    c = [1]*n
    return np.dot(c, x)

def x3_cost(x):
    # Return an Expression
    c = [1]*n
    return np.dot(c, x)

def x1_x2_cost(x, x_):
    # Return an Expression
    c = [1]*n
    e = -x - x_
    return np.dot(c, e)

def x1_x3_cost(x, x_):
    # Return an Expression
    c = [1]*n
    e = x - x_
    return np.dot(c, e)

def x2_x3_cost(x, x_):
    # Return an Expression
    c = [1]*n
    e = x - x_
    return np.dot(c, e)

def x1_residual(x):
    A = np.array([1])
    b = np.array([4])
    return A @ x - b

def x2_residual(x):
    A = np.array([1])
    b = np.array([4])
    return A @ x - b

def x3_residual(x):
    A = np.array([1])
    b = np.array([-2])
    return A @ x - b

def x1_x2_residual(x, x_):
    A = np.array([1])
    B = np.array([1])
    c = np.array([8])
    return A @ x + B @ x_ - c

def x2_x3_residual(x, x_):
    A = np.array([1])
    B = np.array([1])
    c = np.array([2])
    return A @ x + B @ x_ - c

def x1_x3_residual(x, x_):
    A = np.array([1])
    B = np.array([1])
    c = np.array([2])
    return A @ x + B @ x_ - c


################################################################################
# Complete GCS solved using LP
################################################################################
prog.AddCost(x1_cost(x1))
prog.AddCost(x2_cost(x2))
prog.AddCost(x3_cost(x3))
prog.AddCost(x1_x2_cost(x1, x2))
prog.AddCost(x1_x3_cost(x1, x3))
prog.AddCost(x2_x3_cost(x2, x3))
prog.AddConstraint(x1_residual(x1) == 0)
prog.AddConstraint(x2_residual(x2) == 0)
prog.AddConstraint(x3_residual(x3) == 0)
prog.AddConstraint(x1_x2_residual(x1, x2) == 0)
prog.AddConstraint(x2_x3_residual(x2, x3) == 0)
prog.AddConstraint(x1_x3_residual(x1, x3) == 0)


result = Solve(prog)

print(f"Is solved successfully: {result.is_success()}")
print(f"x1 optimal value: {result.GetSolution(x1)}")
print(f"x2 optimal value: {result.GetSolution(x2)}")
print(f"x3 optimal value: {result.GetSolution(x3)}")
print(f"optimal cost: {result.get_optimal_cost()}")





################################################################################
# GCS solved with ADMM
################################################################################
m_total = x1_residual() # total number of constraints in the entire program

# Initialize variables to 0
x_1, x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3, y = 0

# Symbolic Expression for Lagrangian
x_1_sym = np.array([Variable("x_1")])
x_2_sym = np.array([Variable("x_2")])
x_3_sym = np.array([Variable("x_3")])
z1_1_sym = np.array([Variable("z1_1")])
z1_2_sym = np.array([Variable("z1_2")])
z2_2_sym = np.array([Variable("z2_2")])
z2_3_sym = np.array([Variable("z2_3")])
z3_1_sym = np.array([Variable("z3_1")])
z3_3_sym = np.array([Variable("z3_3")])
y_sym = np.array([Variable("y")])

def augmented_lagrangian(x_1_sym, x_2_sym, x_3_sym, z1_1_sym, z1_2_sym, z2_2_sym, z2_3_sym, z3_1_sym, z3_3_sym, y):
    return x1_cost(x_1_sym) + x2_cost(x_2_sym) + x3_cost(x_3_sym) + x1_x2_cost(z1_1_sym, z1_2_sym) + x2_x3_cost(z2_2_sym, z2_3_sym) + x1_x3_cost(z3_1_sym, z3_3_sym) \
        + np.dot(y, ())
    pass
    

def argmin_x1():
    pass

def argmin_x2():
    pass

def argmin_x3():
    pass

def argmin_z1_1():
    pass

def argmin_z1_2():
    pass

def argmin_z2_2():
    pass

def argmin_z2_3():
    pass

def argmin_z3_1():
    pass

def argmin_z3_3():
    pass

iter_limit = 100
for i in range(iter_limit):
    x_1 = argmin_x1(x_2, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3)
    
    x_2 = argmin_x1(x_1, x_3, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3)
    
    x_3 = argmin_x1(x_1, x_2, z1_1, z1_2, z2_2, z2_3, z3_1, z3_3)