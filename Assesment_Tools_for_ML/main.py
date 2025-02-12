import gurobipy as gp
from gurobipy import GRB
import pulp
import matplotlib.pyplot as plt
import time
import data as d
from viz import plot_comparison, plot_schedule


CPLEX_PATH = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe"
SCIP_PATH = r"C:\Program Files\SCIPOptSuite 9.2.1\bin\scip.exe"


# Define jobs and machines
jobs = d.JOBS
machines = d.MACHINES  # Machines M1 to M8

# Processing times: (job, operation) -> [(machine, time)]
processing_times = d.PROCESSING_TIME

# ====================== GUROBI SOLVER ======================
def solve_with_gurobi():
    """Solve FJSSP using Gurobi's MILP formulation"""
    start_time = time.time()
    model = gp.Model("FJSSP")

    # ========== DECISION VARIABLES ==========
    # Assignment variables: x[job, op, machine] = 1 if operation assigned to machine
    x = model.addVars(
        [(job, op, m) for (job, op) in processing_times 
        for m, _ in processing_times[job, op]],
        vtype=GRB.BINARY, name="x_assign"
    )
    
    # Start time variables: start[job, op] = operation start time
    start = model.addVars(
        processing_times.keys(), 
        vtype=GRB.INTEGER, 
        name="start_time"
    )
    
    # Makespan variable: Maximum completion time across all operations
    C_max = model.addVar(vtype=GRB.INTEGER, name="makespan")

    # ========== CONSTRAINTS ==========
    # 1. Operation Assignment Constraint:
    # Each operation must be assigned to exactly one machine
    # ∑ x[job,op,m] = 1 ∀ (job,op)
    for (job, op) in processing_times:
        model.addConstr(
            sum(x[job, op, m] for m, _ in processing_times[job, op]) == 1,
            name=f"Assign_{job}_{op}"
        )

    # 2. Precedence Constraint:
    # Operations must follow job sequence
    # start[job,op_{k+1}] ≥ start[job,op_k} + processing_time(op_k)
    for job in jobs:
        # Get sorted operations for current job
        operations = sorted(op for (j, op) in processing_times if j == job)
        for i in range(len(operations)-1):
            op1, op2 = operations[i], operations[i+1]
            # Calculate processing time as linear expression
            processing_time = gp.quicksum(
                x[job, op1, m] * t 
                for m, t in processing_times[job, op1]
            )
            model.addConstr(
                start[job, op2] >= start[job, op1] + processing_time,
                name=f"Precedence_{job}_{op1}_{op2}"
            )

    # 3. No-Overlap Constraint (Disjunctive):
    # Operations on same machine cannot overlap
    # Uses Big-M method with binary sequencing variable
    M = 10000  # Large enough constant (greater than max possible makespan)
    for m in machines:
        # Get all operations that can use machine m
        machine_ops = [(job, op) for (job, op) in processing_times 
                      if m in [machine for machine, _ in processing_times[job, op]]]
        
        # Create constraints for all operation pairs
        for i in range(len(machine_ops)):
            for j in range(i+1, len(machine_ops)):
                job1, op1 = machine_ops[i]
                job2, op2 = machine_ops[j]
                
                # Sequencing variable: 0=op1 first, 1=op2 first
                y = model.addVar(vtype=GRB.BINARY, name=f"seq_{m}_{i}_{j}")
                
                # Processing times on machine m
                p1 = sum(x[job1, op1, m] * t for m_, t in processing_times[job1, op1] if m_ == m)
                p2 = sum(x[job2, op2, m] * t for m_, t in processing_times[job2, op2] if m_ == m)
                
                # Big-M constraints for both sequencing possibilities
                # If y=0: op1 must finish before op2 starts
                model.addConstr(
                    start[job1, op1] + p1 <= start[job2, op2] + M*(1 - y),
                    name=f"Disj_{m}_{i}_{j}_1"
                )
                # If y=1: op2 must finish before op1 starts
                model.addConstr(
                    start[job2, op2] + p2 <= start[job1, op1] + M*y,
                    name=f"Disj_{m}_{i}_{j}_2"
                )

    # 4. Makespan Constraint:
    # C_max ≥ completion time of all operations
    # C_max ≥ start[job,op] + processing_time ∀ (job,op)
    for (job, op) in processing_times:
        model.addConstr(
            C_max >= start[job, op] + gp.quicksum(
                x[job, op, m] * t for m, t in processing_times[job, op]
            ),
            name=f"Makespan_{job}_{op}"
        )

    # ========== SOLVE & RESULTS ==========
    model.setObjective(C_max, GRB.MINIMIZE)
    model.optimize()

    # Process results
    if model.status == GRB.OPTIMAL:
        return {
            'makespan': C_max.X,
            'assignments': {(job, op): m for (job, op) in processing_times 
                           for m, _ in processing_times[job, op] if x[job, op, m].X > 0.5},
            'start_times': {k: v.X for k, v in start.items()},
            'solve_time': time.time() - start_time
        }
    else:
        raise RuntimeError("Gurobi failed to find optimal solution")



# Helper function to get operations on a machine
def operations_on_m(machine):
    return [(job, op) for (job, op) in processing_times 
            if any(m == machine for m, _ in processing_times[(job, op)])]


# PuLP/CPLEX solver implementation
def solve_with_pulp(execution='CPLEX'):
    start_time = time.time()  # Record start time

    # Create PuLP model
    problem = pulp.LpProblem("FJSP", pulp.LpMinimize)

    # Decision Variables
    # Assignment variables (x[job,op,machine])
    x = pulp.LpVariable.dicts(
        "x", 
        [(job, op, m) for (job, op) in processing_times 
        for m, _ in processing_times[(job, op)]], 
        cat='Binary'
    )

    # Start time variables
    start = pulp.LpVariable.dicts(
        "start", 
        processing_times.keys(), 
        lowBound=0,
        cat='Integer'
    )

    # Makespan variable
    C_max = pulp.LpVariable("C_max", lowBound=0, cat='Integer')

    # Objective function
    problem += C_max

    # ========== CONSTRAINTS ==========
    # 1. Assignment constraints
    for (job, op) in processing_times:
        problem += pulp.lpSum(
            x[(job, op, m)] for m, _ in processing_times[(job, op)]
        ) == 1, f"Assign_{job}_{op}"

    # 2. Precedence constraints
    for job in jobs:
        operations = [op for (j, op) in processing_times if j == job]
        for i in range(len(operations)-1):
            op1 = operations[i]
            op2 = operations[i+1]
            # Calculate processing time for op1
            processing_time = pulp.lpSum(
                x[(job, op1, m)] * t 
                for m, t in processing_times[(job, op1)]
            )
            problem += (
                start[(job, op2)] >= start[(job, op1)] + processing_time,
                f"Precedence_{job}_{op1}_{op2}"
            )

    # 3. No-overlap constraints
    M = 10000  # Large constant
    for m in machines:
        ops = operations_on_m(m)
        for i in range(len(ops)):
            for j in range(i+1, len(ops)):
                job1, op1 = ops[i]
                job2, op2 = ops[j]
                
                # Get processing times on this machine
                p1 = next(t for machine, t in processing_times[(job1, op1)] if machine == m)
                p2 = next(t for machine, t in processing_times[(job2, op2)] if machine == m)
                
                # Sequencing variable
                y = pulp.LpVariable(
                    f"y_{m}_{i}_{j}", 
                    cat='Binary'
                )
                
                # Big-M constraints
                problem += (
                    start[(job2, op2)] >= start[(job1, op1)] + p1 * x[(job1, op1, m)] - M*(1 - y),
                    f"NoOverlap1_{m}_{i}_{j}"
                )
                problem += (
                    start[(job1, op1)] >= start[(job2, op2)] + p2 * x[(job2, op2, m)] - M*y,
                    f"NoOverlap2_{m}_{i}_{j}"
                )

    # 4. Makespan constraint
    for (job, op) in processing_times:
        processing_time = pulp.lpSum(
            x[(job, op, m)] * t 
            for m, t in processing_times[(job, op)]
        )
        problem += (
            C_max >= start[(job, op)] + processing_time,
            f"Makespan_{job}_{op}"
        )

    if execution == 'CPLEX':
        # Solve with CPLEX
        problem.solve(pulp.CPLEX_CMD(path=CPLEX_PATH,timeLimit=300)) #,timeLimit=300, msg=True,))
    elif execution == 'SCIP':
        problem.solve(pulp.SCIP_CMD(path=SCIP_PATH,timeLimit=300))

    # ========== SOLVE & RESULTS ==========
    if pulp.LpStatus[problem.status] == 'Optimal':
        print(f"Optimal makespan: {C_max.varValue}")
        print("Schedule:")
        for (job, op) in processing_times:
            for m, t in processing_times[(job, op)]:
                if x[(job, op, m)].varValue > 0.5:
                    print(f"{job} Op{op}: Machine {m} (Start: {start[(job, op)].varValue}, Duration: {t})")
    else:
        print("No optimal solution found")
    return {
        'makespan': C_max.varValue,
        'assignments': {(job, op): m for (job, op) in processing_times 
                       for m, _ in processing_times[(job, op)] if x[(job, op, m)].varValue > 0.5},
        'start_times': {k: v.varValue for k, v in start.items()},
        'solve_time': time.time() - start_time
    }

# Main execution
if __name__ == "__main__":
    # Solve with both solvers
    print("Solving with Gurobi...")
    gurobi_results = solve_with_gurobi()
    print("\nSolving with CPLEX...")
    cplex_results = solve_with_pulp()
    print("\nSolving with SCIP...")
    scip_results = solve_with_pulp(execution= 'SCIP')
    
    # Plot results
    plot_schedule("Gurobi", gurobi_results['assignments'], gurobi_results['start_times'])
    plot_schedule("CPLEX", cplex_results['assignments'], cplex_results['start_times'])
    plot_schedule("SCIP", scip_results['assignments'], scip_results['start_times'])
    plot_comparison(gurobi_results, cplex_results, scip_results)
    
    # Show all plots
    plt.show()