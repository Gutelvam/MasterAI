import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import data as d


# Define jobs and machines
jobs = d.JOBS
machines = d.MACHINES  # Machines M1 to M8
# Processing times: (job, operation) -> [(machine, time)]
processing_times = d.PROCESSING_TIME

def generate_distinct_colors(n):
    if n <= 20:
        cmap = cm.tab20
    else:
        rgb_tuples = [(i/n, (i*37%100)/100, (i*61%100)/100) for i in range(n)]
        cmap = mcolors.ListedColormap(rgb_tuples)
    return [cmap(i) for i in np.linspace(0, 1, n)]


# Visualization functions
def plot_schedule(solver_name, assignments, start_times):
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = generate_distinct_colors(len(jobs))
    job_to_color = {job: colors[idx] for idx, job in enumerate(jobs)}

    # Correctly calculate makespan:
    makespan = 0
    for (job, op), m in assignments.items():
        t = next((t for machine, t in processing_times.get((job, op), []) if machine == m), None) # Handle missing key
        if t is None:  # Check if operation exists
            print(f"Warning: Operation {op} for job {job} not found in processing times.")
            continue  # Skip to the next iteration

        start_time = start_times[(job, op)]
        makespan = max(makespan, start_time + t)  # Update makespan

        ax.barh(y=m, width=t, left=start_time, height=0.8,
                color=job_to_color[job], edgecolor='black')
        ax.text(start_time + t / 2, m, f"OP{op}",
                ha='center', va='center', color='white', fontsize=8)

    ax.set_yticks(machines)
    ax.set_yticklabels([f'M{m}' for m in machines])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(f'{solver_name} Schedule (Makespan: {makespan})')  # Use calculated makespan

    legend_elements = [patches.Patch(color=color, label=job)
                       for job, color in job_to_color.items()]
    ax.legend(handles=legend_elements, title="Jobs",
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_comparison_two_diff(gurobi_results, cplex_results, scip_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Makespan comparison
    ax1.bar(['Gurobi', 'CPLEX','SCIP'], 
            [gurobi_results['makespan'], cplex_results['makespan'],scip_results['makespan']], 
            color=['blue', 'orange', 'green'])
    ax1.set_title('Makespan Comparison')
    ax1.set_ylabel('Time Units')
    
    # Solve time comparison
    ax2.bar(['Gurobi', 'CPLEX','SCIP'], 
            [gurobi_results['solve_time'], cplex_results['solve_time'],scip_results['solve_time']], 
            color=['blue', 'orange', 'green'])
    ax2.set_title('Solve Time Comparison')
    ax2.set_ylabel('Seconds')
    
    plt.tight_layout()
    return fig

def plot_comparison(gurobi_results, cplex_results, scip_results):
    fig, ax1 = plt.subplots(figsize=(8, 6))  # Single subplot

    # Makespan bars
    makespans = [gurobi_results['makespan'], cplex_results['makespan'], scip_results['makespan']]
    solver_names = ['Gurobi', 'CPLEX', 'SCIP']
    bar_width = 0.35  # Adjust width for better visualization
    bar_positions = [i for i in range(len(solver_names))] # positions for the bars

    ax1.bar(bar_positions, makespans, bar_width, color=['blue', 'orange', 'green'], label='Makespan')

    # Solve time line
    solve_times = [gurobi_results['solve_time'], cplex_results['solve_time'], scip_results['solve_time']]
    # Offset the line slightly to the right of the bars for better visibility
    line_positions = [i + bar_width/2 for i in range(len(solver_names))] # positions for the line
    ax1.plot(line_positions, solve_times, color='red', marker='o', linestyle='-', label='Solve Time')

    # Set labels and title
    ax1.set_xticks([i + bar_width/2 for i in range(len(solver_names))]) #set x ticks in the middle of the bars
    ax1.set_xticklabels(solver_names) # set the labels of the x ticks
    ax1.set_ylabel('Value')  # Combined y-axis label
    ax1.set_title('Performance Comparison')

    # Add legend
    ax1.legend()

    # Improve layout
    plt.tight_layout()
    return fig