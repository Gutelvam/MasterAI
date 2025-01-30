from gurobipy import Model, GRB
import matplotlib.pyplot as plt

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
model = Model("FJSP")

# Variables
X = {}  # X[j, o, m] = 1 if operation o of job j is assigned to machine m
S = {}  # S[j, o] = start time of operation o of job j
C = {}  # Completion time of operation o of job j

for j, ops in jobs.items():
    for o, (m, p) in enumerate(ops):
        for machine in machines:
            X[j, o, machine] = model.addVar(vtype=GRB.BINARY, name=f"X_{j}_{o}_{machine}")
        S[j, o] = model.addVar(vtype=GRB.CONTINUOUS, name=f"S_{j}_{o}")
        C[j, o] = model.addVar(vtype=GRB.CONTINUOUS, name=f"C_{j}_{o}")

# Constraints
for j, ops in jobs.items():
    for o, (m, p) in enumerate(ops):
        # Assign each operation to exactly one machine
        model.addConstr(sum(X[j, o, machine] for machine in machines) == 1)

        # Define completion time
        model.addConstr(C[j, o] == S[j, o] + sum(p * X[j, o, machine] for machine in machines))

        # Precedence constraints (each operation starts after the previous one finishes)
        if o > 0:
            model.addConstr(S[j, o] >= C[j, o - 1])

# No overlapping of operations on the same machine
big_M = 1e6  # Large constant to enforce logical conditions
for m in machines:
    for j1, ops1 in jobs.items():
        for j2, ops2 in jobs.items():
            if j1 != j2:
                for o1 in range(len(ops1)):
                    for o2 in range(len(ops2)):
                        if o1 != o2:
                            Y = model.addVar(vtype=GRB.BINARY, name=f"Y_{j1}_{o1}_{j2}_{o2}")

                            # Enforce non-overlapping conditions using Y
                            model.addConstr(S[j1, o1] >= C[j2, o2] - big_M * (1 - Y))
                            model.addConstr(S[j2, o2] >= C[j1, o1] - big_M * Y)


# Objective: Minimize makespan (max completion time)
makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")
for j, ops in jobs.items():
    for o in range(len(ops)):
        model.addConstr(makespan >= C[j, o])

model.setObjective(makespan, GRB.MINIMIZE)

# Solve Model
model.optimize()

# Extract solution
solution = {}
for j, ops in jobs.items():
    for o in range(len(ops)):
        for m in machines:
            if X[j, o, m].X > 0.5:  # If assigned
                solution[(j, o)] = (m, S[j, o].X, C[j, o].X)

# Generate Gantt Chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
job_colors = {j: colors[i % len(colors)] for i, j in enumerate(jobs)}

for (j, o), (m, start, end) in solution.items():
    ax.barh(m, end - start, left=start, color=job_colors[j], edgecolor="black", label=j if o == 0 else "")

ax.set_xlabel("Time")
ax.set_ylabel("Machines")
ax.set_title("FJSP Gantt Chart")
ax.legend(loc="upper right")
plt.grid()
plt.show()
