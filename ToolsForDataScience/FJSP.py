import gurobipy as gp
from gurobipy import GRB
from pyomo.environ import *
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
from gekko import GEKKO
import matplotlib.pyplot as plt
import numpy as np

# Problem Parameters
jobs = {
    "pr1,2": [(1, 3), (2, 4), (3, 5), (4, 5)],
    "pr2,2": [(1, 5), (2, 4), (3, 6), (4, 4)],
    "pr3,2": [(2, 3), (3, 8), (4, 5), (5, 7)],
    "pr4,2": [(3, 7), (4, 7), (5, 8), (6, 7)],
    "pr5,1": [(1, 3), (2, 5), (3, 6)],
    "pr6,3": [(1, 2), (3, 4), (4, 5)],
    "pr7,2": [(3, 5), (4, 8), (5, 6)],
    "pr8,1": [(2, 6), (4, 5), (5, 8)]
}

machines = {1, 2, 3, 4, 5, 6, 7, 8}  # Set of machines

# Gurobi Model
def solve_gurobi():
    model = gp.Model("FJSP")
    S = model.addVars(jobs.keys(), range(4), vtype=GRB.CONTINUOUS, name="Start")
    C = model.addVars(jobs.keys(), range(4), vtype=GRB.CONTINUOUS, name="Completion")
    X = model.addVars(jobs.keys(), range(4), machines, vtype=GRB.BINARY, name="MachineSelection")
    
    for j, ops in jobs.items():
        for o, (m, p) in enumerate(ops):
            model.addConstr(gp.quicksum(X[j, o, k] for k in machines) == 1)
            model.addConstr(C[j, o] == S[j, o] + gp.quicksum(p * X[j, o, k] for k in machines))
            if o > 0:
                model.addConstr(S[j, o] >= C[j, o - 1])
    
    model.setObjective(gp.quicksum(C[j, len(ops)-1] for j, ops in jobs.items()), GRB.MINIMIZE)
    model.optimize()
    return model

# Pyomo Model
def solve_pyomo():
    model = ConcreteModel()
    model.S = Var(jobs.keys(), range(4), domain=NonNegativeReals)
    model.C = Var(jobs.keys(), range(4), domain=NonNegativeReals)
    model.X = Var(jobs.keys(), range(4), machines, domain=Binary)
    
    def obj_rule(m):
        return sum(m.C[j, len(ops)-1] for j, ops in jobs.items())
    
    model.obj = Objective(rule=obj_rule, sense=minimize)
    solver = SolverFactory('glpk')
    solver.solve(model)
    return model

# PuLP Model
def solve_pulp():
    model = LpProblem("FJSP", LpMinimize)
    S = { (j, o): LpVariable(f"S_{j}_{o}", lowBound=0) for j in jobs for o in range(4) }
    C = { (j, o): LpVariable(f"C_{j}_{o}", lowBound=0) for j in jobs for o in range(4) }
    X = { (j, o, k): LpVariable(f"X_{j}_{o}_{k}", cat="Binary") for j in jobs for o in range(4) for k in machines }
    
    model += lpSum(C[j, len(ops)-1] for j, ops in jobs.items())
    solver = PULP_CBC_CMD()
    model.solve(solver)
    return model

# GEKKO Model
def solve_gekko():
    m = GEKKO(remote=False)
    S = m.Array(m.Var, (len(jobs), 4), lb=0)
    C = m.Array(m.Var, (len(jobs), 4), lb=0)
    X = m.Array(m.Var, (len(jobs), 4, len(machines)), lb=0, ub=1, integer=True)
    
    m.Obj(sum(C[j][len(ops)-1] for j, ops in enumerate(jobs.values())))
    m.solve(disp=False)
    return m

# Gantt Chart Visualization
def plot_gantt(schedule):
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(jobs)))
    for j, ops in schedule.items():
        for o, (m, start, end) in enumerate(ops):
            ax.barh(m, end - start, left=start, color=colors[o % len(colors)], edgecolor='black')
    plt.xlabel("Time")
    plt.ylabel("Machines")
    plt.title("FJSP Gantt Chart")
    plt.show()

# Run and Compare Solutions
models = {
    "Gurobi": solve_gurobi(),
    "Pyomo": solve_pyomo(),
    "PuLP": solve_pulp(),
    "GEKKO": solve_gekko()
}
plot_gantt(models["Gurobi"])
