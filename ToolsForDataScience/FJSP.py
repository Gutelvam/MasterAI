import gurobipy as gp
from gurobipy import GRB

# Define jobs and operations based on the provided process plan
jobs = [
    [(1, 3), (2, 4), (3, 5), (4, 5)],  
    [(1, 3, 5), (4, 8), (4, 6), (4, 7)],  
    [(2, 3, 8), (4, 8), (3, 5, 7), (4, 6)],  
    [(1, 3, 5), (2, 8), (3, 4, 6, 7), (5, 6, 8)],  
    [(1,), (2, 4), (3, 8), (5, 6, 8)],  
    [(1, 2, 3), (4, 5), (3, 6)],  
    [(3, 5, 6), (4, 7, 8), (1, 3, 4, 5), (4, 6, 8)],  
    [(1, 2, 6), (4, 5, 8), (3, 7), (4, 6)]  
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

# Create Gurobi model
model = gp.Model("FJSP")

# Decision variables
x = model.addVars(num_jobs, 7, num_machines, vtype=GRB.BINARY, name="x")  # Operation assignment
start = model.addVars(num_jobs, 7, vtype=GRB.CONTINUOUS, name="start")  # Operation start times
c_max = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")  # Makespan

# Constraints
for j in range(num_jobs):
    for o in range(len(jobs[j])):
        machines = jobs[j][o]
        model.addConstr(gp.quicksum(x[j, o, m] for m in machines) == 1)  # Each operation assigned to exactly one machine

# Precedence constraints
for j in range(num_jobs):
    for o in range(len(jobs[j]) - 1):
        prev_machines = jobs[j][o]
        next_machines = jobs[j][o + 1]
        for i, pm in enumerate(prev_machines):
            for k, nm in enumerate(next_machines):
                model.addConstr(
                    start[j, o + 1] >= start[j, o] + x[j, o, pm] * processing_times[j][o][i]
                )  # Ensure next operation starts after the previous one finishes

# Non-overlapping constraint for machines
for m in range(1, num_machines + 1):
    for j1 in range(num_jobs):
        for o1 in range(len(jobs[j1])):
            for j2 in range(num_jobs):
                if j1 != j2:
                    for o2 in range(len(jobs[j2])):
                        model.addConstr(
                            start[j2, o2] >= start[j1, o1] + gp.quicksum(
                                x[j1, o1, m] * processing_times[j1][o1][i] for i in range(len(jobs[j1][o1]))
                            ) - (1 - x[j1, o1, m]) * 10000
                        )  # Enforce non-overlap

# Makespan definition
for j in range(num_jobs):
    for o in range(len(jobs[j])):
        machines = jobs[j][o]
        for i, m in enumerate(machines):
            model.addConstr(
                c_max >= start[j, o] + x[j, o, m] * processing_times[j][o][i]
            )  # Ensure makespan is maximum completion time

# Objective: Minimize makespan
model.setObjective(c_max, GRB.MINIMIZE)

# Solve model
model.optimize()

# Output results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Makespan: {c_max.X}")
    for j in range(num_jobs):
        for o in range(len(jobs[j])):
            for m in range(1, num_machines + 1):
                if x[j, o, m].X > 0.5:
                    print(f"Job {j+1}, Operation {o+1} assigned to Machine {m} starting at {start[j, o].X}")

