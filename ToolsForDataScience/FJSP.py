import time
import random
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, Binary
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, PULP_CBC_CMD
from gekko import GEKKO

# Define Problem Parameters
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
machines = {1, 2, 3, 4, 5, 6, 7, 8}

# Solve using Gurobi
def solve_gurobi():
    model = Model("FJSP")
    X, S, C = {}, {}, {}

    for j, ops in jobs.items():
        for o, (m, p) in enumerate(ops):
            for machine in machines:
                X[j, o, machine] = model.addVar(vtype=GRB.BINARY)
            S[j, o] = model.addVar(vtype=GRB.CONTINUOUS)
            C[j, o] = model.addVar(vtype=GRB.CONTINUOUS)

    for j, ops in jobs.items():
        for o, (m, p) in enumerate(ops):
            model.addConstr(sum(X[j, o, machine] for machine in machines) == 1)
            model.addConstr(C[j, o] == S[j, o] + sum(p * X[j, o, machine] for machine in machines))
            if o > 0:
                model.addConstr(S[j, o] >= C[j, o - 1])

    makespan = model.addVar(vtype=GRB.CONTINUOUS)
    for j, ops in jobs.items():
        for o in range(len(ops)):
            model.addConstr(makespan >= C[j, o])

    model.setObjective(makespan, GRB.MINIMIZE)
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    solution = {(j, o): (m, S[j, o].X, C[j, o].X)
                for j, ops in jobs.items()
                for o, (m, p) in enumerate(ops)
                for m in machines if X[j, o, m].X > 0.5}

    return solution, solve_time, makespan.X

# Solve using Pyomo
def solve_pyomo():
    model = ConcreteModel()
    model.S = Var(jobs.keys(), range(4), within=NonNegativeReals)
    model.C = Var(jobs.keys(), range(4), within=NonNegativeReals)
    model.makespan = Var(within=NonNegativeReals)

    def obj_rule(model):
        return model.makespan
    model.obj = Objective(rule=obj_rule)

    solver = SolverFactory("gurobi")
    start_time = time.time()
    solver.solve(model)
    solve_time = time.time() - start_time

    return {}, solve_time, model.makespan.value

# Solve using PuLP
def solve_pulp():
    prob = LpProblem("FJSP", LpMinimize)
    makespan = LpVariable("Makespan", lowBound=0)
    prob += makespan

    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=False))
    solve_time = time.time() - start_time

    return {}, solve_time, makespan.varValue

# Solve using GEKKO
def solve_gekko():
    m = GEKKO(remote=False)
    makespan = m.Var(lb=0)
    m.Obj(makespan)
    
    start_time = time.time()
    m.solve(disp=False)
    solve_time = time.time() - start_time

    return {}, solve_time, makespan.value

# Function to plot Gantt chart
def plot_gantt_chart(solution, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    job_colors = {j: (random.random(), random.random(), random.random()) for j in jobs}

    for (j, o), (m, start, end) in solution.items():
        ax.barh(m, end - start, left=start, color=job_colors[j], edgecolor="black", label=j if o == 0 else "")

    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.grid()
    plt.show()

# Solve problems
solution_gurobi, time_gurobi, makespan_gurobi = solve_gurobi()
solution_pyomo, time_pyomo, makespan_pyomo = solve_pyomo()
solution_pulp, time_pulp, makespan_pulp = solve_pulp()
solution_gekko, time_gekko, makespan_gekko = solve_gekko()

# Plot Gantt charts
plot_gantt_chart(solution_gurobi, "Gantt Chart (Gurobi)")
plot_gantt_chart(solution_pyomo, "Gantt Chart (Pyomo)")
plot_gantt_chart(solution_pulp, "Gantt Chart (PuLP)")
plot_gantt_chart(solution_gekko, "Gantt Chart (GEKKO)")

# Performance Comparison
solvers = ["Gurobi", "Pyomo", "PuLP", "GEKKO"]
solve_times = [time_gurobi, time_pyomo, time_pulp, time_gekko]
makespans = [makespan_gurobi, makespan_pyomo, makespan_pulp, makespan_gekko]

# Bar Chart for Solver Comparison
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(solvers))
width = 0.35
ax.bar(x - width/2, solve_times, width, label="Solve Time (s)")
ax.bar(x + width/2, makespans, width, label="Makespan")
ax.set_xticks(x)
ax.set_xticklabels(solvers)
ax.set_xlabel("Solvers")
ax.set_title("Solver Performance Comparison")
ax.legend()
plt.show()
