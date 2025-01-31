import gurobipy as gp
from gurobipy import GRB
from pyomo.environ import *
import matplotlib.pyplot as plt
import pandas as pd

# Data from the process plan (Corrected data for pr7,2)
process_plan = {
    'pr1,2': {1: [(1, 3), (3, 5)], 2: [(2, 4), (4, 5)], 3: [(3, 5)], 4: [(4, 5), (5, 5), (6, 4), (7, 5), (8, 9)]},
    'pr2,2': {1: [(1, 5), (7, 7)], 2: [(4, 8)], 3: [(4, 6)], 4: [(4, 4), (7, 7)], 5: [(1, 2)], 6: [(5, 6), (4, 4)], 7: [(4, 4)]},
    'pr3,3': {1: [(7, 6), (8, 8)], 2: [(7, 7)], 3: [(7, 8), (7, 7)], 4: [(7, 8)], 5: [(1, 4)]},
    'pr4,2': {1: [(1, 3), (5, 5)], 2: [(4, 8), (2, 8)], 3: [(3, 4), (6, 7)], 4: [(4, 4)], 5: [(4, 5), (6, 7)], 6: [(3, 5), (5, 5)]},
    'pr5,1': {1: [(1, 1)], 2: [(2, 4), (4, 5)], 3: [(3, 8)], 4: [(5, 6), (8, 8)], 5: [(4, 6)]},
    'pr6,3': {1: [(3, 3)], 2: [(4, 5)], 3: [(4, 4)], 4: [(3, 3), (3, 3), (3, 3)], 5: [(5, 4)]},
    'pr7,2': {1: [(3, 5), (6, 6)], 2: [(7, 8)], 3: [(9, 8)], 4: [(4, 6)], 5: [(4, 6)], 6: [(3, 3)]},  # Corrected
    'pr8,1': {1: [(3, 5), (6, 4)], 2: [(4, 6), (4, 4)], 3: [(3, 7), (4, 5)], 4: [(4, 6), (4, 6)], 5: [(1, 2)]}
}

jobs = list(process_plan.keys())
machines = sorted(list(set(m for job_data in process_plan.values() for op_data in job_data.values() for m, _ in op_data)))
num_jobs = len(jobs)
num_machines = len(machines)

# --- Gurobi Model ---
model_gurobi = gp.Model("FJSP_Gurobi")
start_times_gurobi = model_gurobi.addVars(jobs, range(1, 8), machines, vtype=GRB.CONTINUOUS, name="start")
assigned_gurobi = model_gurobi.addVars(jobs, range(1, 8), machines, vtype=GRB.BINARY, name="assigned")
makespan_gurobi = model_gurobi.addVar(vtype=GRB.CONTINUOUS, name="makespan")

model_gurobi.setObjective(makespan_gurobi, GRB.MINIMIZE)


# Constraints (Gurobi)
for job in jobs:
    for op in process_plan[job]:
        model_gurobi.addConstr(sum(assigned_gurobi[job, op, m] for m in machines if any(mach == m for mach, _ in process_plan[job][op])) == 1)

for job in jobs:
    ops = sorted(process_plan[job].keys())
    for i in range(len(ops) - 1):
        op1 = ops[i]
        op2 = ops[i + 1]
        for m in machines if any(mach == m for mach, _ in process_plan[job][op1]):
            for n in machines if any(mach == n for mach, _ in process_plan[job][op2]):
                duration1 = [d for mach, d in process_plan[job][op1] if mach == m][0] if any(mach == m for mach, _ in process_plan[job][op1]) else 0
                model_gurobi.addConstr(start_times_gurobi[job, op2, n] >= start_times_gurobi[job, op1, m] + duration1 - 10000 * (1 - assigned_gurobi[job, op1, m]) - 10000 * (1 - assigned_gurobi[job, op2, n]))

for m in machines:
    for job1 in jobs:
        for op1 in process_plan[job1]:
            for job2 in jobs:
                if job1 != job2:
                    for op2 in process_plan[job2]:
                        duration1 = [d for mach, d in process_plan[job1][op1] if mach == m][0] if any(mach == m for mach, _ in process_plan[job1][op1]) else 0
                        duration2 = [d for mach, d in process_plan[job2][op2] if mach == m][0] if any(mach == m for mach, _ in process_plan[job2][op2]) else 0
                        model_gurobi.addConstr(start_times_gurobi[job1, op1, m] + duration1 <= start_times_gurobi[job2, op2, m] + 10000 * (2 - assigned_gurobi[job1, op1, m] - assigned_gurobi[job2, op2, m]) or start_times_gurobi[job2, op2, m] + duration2 <= start_times_gurobi[job1, op1, m] + 10000 * (2 - assigned_gurobi[job1, op1, m] - assigned_gurobi[job2, op2, m]))

for job in jobs:
    for op in process_plan[job]:
        for m in machines if any(mach == m for mach, _ in process_plan[job][op]):
            duration = [d for mach, d in process_plan[job][op] if mach == m][0] if any(mach == m for mach, _ in process_plan[job][op]) else 0
            model_gurobi.addConstr(makespan_gurobi >= start_times_gurobi[job, op, m] + duration)


model_gurobi.optimize()

# --- Pyomo Model ---
model_pyomo = ConcreteModel()
model_pyomo.JOBS = Set(initialize=jobs)
model_pyomo.MACHINES = Set(initialize=machines)
model_pyomo.OPERATIONS = Set(initialize=range(1, 8))

model_pyomo.start = Var(model_pyomo.JOBS, model_pyomo.OPERATIONS, model_pyomo.MACHINES, within=NonNegativeReals)
model_pyomo.assign = Var(model_pyomo.JOBS, model_pyomo.OPERATIONS, model_pyomo.MACHINES, within=Binary)
model_pyomo.makespan = Var(within=NonNegativeReals)

model_pyomo.objective = Objective(expr=model_pyomo.makespan, sense=minimize)

# Constraints (Pyomo)
def rule_assign_pyomo(model, j, o):
    return sum(model.assign[j, o, m] for m in model.MACHINES if any(mach == m for mach, _ in process_plan[j][o])) == 1
model_pyomo.AssignConstraint = Constraint(model_pyomo.JOBS, model_pyomo.OPERATIONS, rule=rule_assign_pyomo)


def rule_precedence_pyomo(model, j, o1, o2):
    if o2 > o1: # Only apply if o2 is after o1
      for m in model.MACHINES if any(mach == m for mach, _ in process_plan[j][o1]):
          for n in model.MACHINES if any(mach == n for mach, _ in process_plan[j][o2]):
              duration1 = [d for mach, d in process_plan[j][o1] if mach == m][0] if any(mach == m for mach, _ in process_plan[j][o1]) else 0
              return model.start[j, o2, n] >= model.start[j, o1, m] + duration1
    else:
      return Constraint.Skip # Skip if o2 is not after o1

model_pyomo.PrecedenceConstraint = Constraint(model_pyomo.JOBS, model_pyomo.OPERATIONS, model_pyomo.OPERATIONS, rule=rule_precedence_pyomo)


def rule_machine_capacity_pyomo(model, m):
    # Initialize an expression for the constraint
    expr = 0
    for j in model.JOBS:
        for o in model.OPERATIONS:
            duration = [d for mach, d in process_plan[j][o] if mach == m][0] if any(mach == m for mach, _ in process_plan[j][o]) else 0
            expr += model.assign[j, o, m] * duration
    return expr <= sum(duration for j in model.JOBS for o in model.OPERATIONS for mach, duration in process_plan[j][o] if mach == m)

model_pyomo.MachineCapacityConstraint = Constraint(model_pyomo.MACHINES, rule=rule_machine_capacity_pyomo)


def rule_makespan_pyomo(model):
    return model.makespan >= max(model.start[j, o, m] + [d for mach, d in process_plan[j][o] if mach == m][0] if any(mach == m for mach, _ in process_plan[j][o]) else 0 for j in model.JOBS for o in model.OPERATIONS for m in model.MACHINES if any(mach == m for mach, _ in process_plan[j][o]))

model_pyomo.MakespanConstraint = Constraint(rule=rule_makespan_pyomo)



# Solve Pyomo model
solver = SolverFactory('glpk')  # Or 'cplex', 'gurobi', etc. if you have them installed
results_pyomo = solver.solve(model_pyomo)

# --- Plotting and Results ---
gurobi_makespan = makespan_gurobi.x if model_gurobi.status == GRB.OPTIMAL else float('inf')
pyomo_makespan = model_pyomo.makespan.value if results_pyomo.solver.termination_condition == TerminationCondition.optimal else float('inf')

print(f"Gurobi Makespan: {gurobi_makespan}")
print(f"Pyomo Makespan: {pyomo_makespan}")

# --- Gantt Chart for Gurobi ---
if model_gurobi.status == GRB.OPTIMAL:
    schedule_data = []
    for job in jobs:
        for op in sorted(process_plan[job].keys()):
            for m in machines:
                if assigned_gurobi[job, op, m].x == 1:
                    duration = [d for mach, d in process_plan[job][op] if mach == m][0] if any(mach == m for mach, _ in process_plan[job][op]) else 0
                    start_time = start_times_gurobi[job, op, m].x
                    schedule_data.append({'Job': job, 'Operation': op, 'Machine': m, 'Start': start_time, 'Duration': duration})

    df = pd.DataFrame(schedule_data)
    df['End'] = df['Start'] + df['Duration']

    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.barh(row['Machine'], row['Duration'], left=row['Start'], height=0.8, label=f"{row['Job']}-{row['Operation']}" if i == 0 else None)

    plt.xlabel("Time")
    plt.ylabel("Machine")
    plt.title("Gurobi Schedule Gantt Chart")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# --- Gantt Chart for Pyomo ---
if results_pyomo.solver.termination_condition == TerminationCondition.optimal:
    schedule_data_pyomo = []
    for j in model_pyomo.JOBS:
        for o in model_pyomo.OPERATIONS:
            for m in model_pyomo.MACHINES:
                if model_pyomo.assign[j, o, m].value == 1:
                    duration = [d for mach, d in process_plan[j][o] if mach == m][0] if any(mach == m for mach, _ in process_plan[j][o]) else 0
                    start_time = model_pyomo.start[j, o, m].value
                    schedule_data_pyomo.append({'Job': j, 'Operation': o, 'Machine': m, 'Start': start_time, 'Duration': duration})

    df_pyomo = pd.DataFrame(schedule_data_pyomo)
    df_pyomo['End'] = df_pyomo['Start'] + df_pyomo['Duration']

    plt.figure(figsize=(10, 6))
    for i, row in df_pyomo.iterrows():
        plt.barh(row['Machine'], row['Duration'], left=row['Start'], height=0.8, label=f"{row['Job']}-{row['Operation']}" if i == 0 else None)

    plt.xlabel("Time")
    plt.ylabel("Machine")
    plt.title("Pyomo Schedule Gantt Chart")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


# --- Performance Comparison Plot ---
makespans = {'Gurobi': gurobi_makespan, 'Pyomo': pyomo_makespan}

plt.figure(figsize=(8, 5))
plt.bar(makespans.keys(), makespans.values(), color=['skyblue', 'lightcoral'])
plt.ylabel("Makespan")
plt.title("Makespan Comparison")

# Add values on top of bars for better readability
for index, value in enumerate(makespans.values()):
    plt.text(index, value, str(round(value, 2)), ha='center', va='bottom')  # Add text above bars

plt.show()




