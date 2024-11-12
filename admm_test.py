from pydrake.solvers import MathematicalProgram, Solve
import numpy as np


# Basic LP
prog = MathematicalProgram()
x = prog.NewContinuousVariables(2)
prog.AddConstraint(x[0] + x[1] == 1)
prog.AddConstraint(x[0] - x[1] <= 0)
prog.AddCost(x[0] * 2 + x[1] * 2)

result = Solve(prog)

print(f"Is solved successfully: {result.is_success()}")
print(f"x optimal value: {result.GetSolution(x)}")
print(f"optimal cost: {result.get_optimal_cost()}")

print("=============================================================")



# ADMM Formulation
max_iters = 100
rho = 0.001  # may need tuning
x_0 = [0, 0]
z_0 = [0, 0]
u_0 = [0, 0]
for i in range(max_iters):
    # Update x
    x_prog = MathematicalProgram()
    x = x_prog.NewContinuousVariables(2)
    x_prog.SetInitialGuess(x, x_0)
    x_prog.AddCost((x[0] * 2 + x[1] * 2) + (rho/2) * (x - z_0 + u_0).T @ (x - z_0 + u_0))
    result = Solve(x_prog)
    print(f"x optimal value: {result.GetSolution(x)}")
    x_0 = result.GetSolution(x)
    
    # Update z (project into feasible set; i.e. find solution in set minimizing Euclidean distance to current x)
    z_prog = MathematicalProgram()
    z = z_prog.NewContinuousVariables(2)
    z_prog.AddConstraint(z[0] + z[1] == 1)
    z_prog.AddConstraint(z[0] - z[1] <= 0)
    z_prog.AddCost((z[0] - x_0[0])**2 + (z[1] - x_0[1])**2)
    result = Solve(z_prog)
    print(f"z optimal value: {result.GetSolution(z)}")
    z_0 = result.GetSolution(z)
    
    # Update u
    u_0 = u_0 + x_0 - z_0
    
    rho = rho * 0.99
    
    
# reduce rho as time progresses
# take answer as average of last 10 iterations