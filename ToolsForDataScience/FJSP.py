import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary
from gekko import GEKKO
import time

# Define data from the provided process plan
jobs = [
    [(1, 3), (2, 4), (3, 5), (4, 5)],  # Job 1
    [(1, 3, 5), (4, 8), (4, 6), (4, 7)],  # Job 2
    [(2, 3, 8), (4, 8), (3, 5, 7), (4, 6)],  # Job 3
    [(1, 3, 5), (2, 8), (3, 4, 6, 7), (5, 6, 8)],  # Job 4
    [(1,), (2, 4), (3, 8), (5, 6, 8)],  # Job 5
    [(1, 2, 3), (4, 5), (3, 6)],  # Job 6
    [(3, 5, 6), (4, 7, 8), (1, 3, 4, 5), (4, 6, 8)],  # Job 7
    [(1, 2, 6), (4, 5, 8), (3, 7), (4, 6)]  # Job 8
]

processing_times = [
    [(4, 5), (4, 5), (5, 6), (5, 5, 4, 5, 9)],
    [(1, 5, 7), (5, 4), (1, 6), (4, 4, 7)],
    [(7, 6, 8), (7, 7), (7, 8, 7), (7, 8)],
    [(4, 3, 7), (4, 4), (4, 5, 6, 7), (3, 5, 5)],
    [(3,), (4, 5), (4, 4), (3, 3, 3)],
    [(3, 5, 6), (7, 8), (9, 8)],
    [(4, 5, 4), (4, 6, 4), (3, 3, 3, 4, 5), (4, 6, 5)],
    [(3, 4, 4), (6, 5, 4), (4, 5), (4, 6)]
]

num_jobs = len(jobs)
num_machines = 8

# Solver Performance Tracking
times = {}

### Gurobi Implementation ###
start_time = time.time()
model = gp.Model("FJSP")

x = model.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name="x")
c = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")

for j in range(num_jobs):
    for o, machines in enumerate(jobs[j]):
        model.addConstr(gp.quicksum(x[j, m] for m in machines) == 1)

# Precedence constraints
for j in range(num_jobs):
    for o in range(len(jobs[j]) - 1):
        model.addConstr(gp.quicksum(x[j, m] * processing_times[j][o][i] for i, m in enumerate(jobs[j][o]))
                        <= gp.quicksum(x[j, m] * processing_times[j][o + 1][i] for i, m in enumerate(jobs[j][o + 1])))

# Makespan constraints
for j in range(num_jobs):
    for o in range(len(jobs[j])):
        model.addConstr(c >= gp.quicksum(x[j, m] * processing_times[j][o][i] for i, m in enumerate(jobs[j][o])))

model.setObjective(c, GRB.MINIMIZE)
model.optimize()
times['Gurobi'] = time.time() - start_time

### Pyomo Implementation ###
start_time = time.time()
pyomo_model = ConcreteModel()
pyomo_model.x = Var(range(num_jobs), range(num_machines), within=Binary)
pyomo_model.makespan = Var(within=NonNegativeReals)

# Objective function
pyomo_model.obj = Objective(expr=pyomo_model.makespan, sense=minimize)
solver = SolverFactory('glpk')
solver.solve(pyomo_model)
times['Pyomo'] = time.time() - start_time

### PuLP Implementation ###
start_time = time.time()
pulp_model = LpProblem("FJSP", LpMinimize)
x_pulp = LpVariable.dicts("x", (range(num_jobs), range(num_machines)), 0, 1, LpBinary)
makespan_pulp = LpVariable("makespan", lowBound=0)

pulp_model += makespan_pulp
pulp_solver = pulp.PULP_CBC_CMD()
pulp_model.solve(pulp_solver)
times['PuLP'] = time.time() - start_time

### Gekko Implementation ###
start_time = time.time()
gekko_model = GEKKO(remote=False)
x_gekko = gekko_model.Array(gekko_model.Var, (num_jobs, num_machines), integer=True, lb=0, ub=1)
makespan_gekko = gekko_model.Var(lb=0)
gekko_model.Obj(makespan_gekko)
gekko_model.solve(disp=False)
times['Gekko'] = time.time() - start_time

# Plot Performance Comparison
plt.figure(figsize=(8, 5))
plt.bar(times.keys(), times.values(), color=['blue', 'red', 'green', 'purple'])
plt.title("Solver Performance Comparison")
plt.xlabel("Solvers")
plt.ylabel("Time (seconds)")
plt.show()
