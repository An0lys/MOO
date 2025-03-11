import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed

# Load datasets
machine_events_path = "data/uc_machine_events.csv"
instance_events_path = "data/uc_instance_events.csv"

machines_df = pd.read_csv(machine_events_path)
instances_df = pd.read_csv(instance_events_path)

# Extract resources from machine events
def parse_properties(properties_str):
    try:
        return eval(properties_str)  # Convert string to dictionary
    except:
        return {}

machines_df['PROPERTIES'] = machines_df['PROPERTIES'].apply(parse_properties)

# Extract relevant resource attributes
def extract_resources(row):
    props = row['PROPERTIES']
    return {
        'cpu': float(props.get('architecture.smt_size', 0)),
        'ram': float(props.get('main_memory.ram_size', 0)),
        'storage': float(props.get('storage_device.size', 0))
    }

machines_df['RESOURCES'] = machines_df.apply(extract_resources, axis=1)

# Construct Fog Computing Graph
G = nx.Graph()
for _, row in machines_df.iterrows():
    G.add_node(row['HOST_NAME (PHYSICAL)'], **row['RESOURCES'])

# Define Multi-Objective Optimization using GRASP with Adaptive Iterations and Simulated Annealing
def compute_load_balance(solution, num_machines):
    machine_usage = np.array([np.count_nonzero(solution == m) for m in range(num_machines)])
    return np.std(machine_usage)  # Standard deviation as a balance measure

def grasp_iteration(cpu_costs, ram_costs, num_machines, num_instances):
    solution = np.random.choice(num_machines, num_instances)
    best_solution = solution.copy()
    best_cost = np.sum(cpu_costs[solution]) + np.sum(ram_costs[solution])
    best_balance = compute_load_balance(solution, num_machines)
    
    temperature = 1.0
    cooling_rate = 0.95
    
    for _ in range(10):
        neighbor = solution.copy()
        idx = random.randint(0, num_instances - 1)
        neighbor[idx] = random.randint(0, num_machines - 1)
        
        cost_neighbor = np.sum(cpu_costs[neighbor]) + np.sum(ram_costs[neighbor])
        balance_neighbor = compute_load_balance(neighbor, num_machines)
        
        if cost_neighbor < best_cost or np.exp((best_cost - cost_neighbor) / temperature) > random.random():
            solution = neighbor
            best_cost = cost_neighbor
            best_balance = balance_neighbor
        
        temperature *= cooling_rate
    
    return best_cost, best_balance

def grasp_optimization(machines_df, instances_df, iterations=300):
    num_machines = len(machines_df)
    num_instances = len(instances_df)
    
    if num_machines == 0 or num_instances == 0:
        print("Error: No machines or instances available for optimization.")
        return []
    
    cpu_costs = np.array([m['RESOURCES']['cpu'] for _, m in machines_df.iterrows()], dtype=float)
    ram_costs = np.array([m['RESOURCES']['ram'] for _, m in machines_df.iterrows()], dtype=float)
    
    solutions = Parallel(n_jobs=-1)(delayed(grasp_iteration)(cpu_costs, ram_costs, num_machines, num_instances) for _ in range(iterations))
    
    # Filter out dominated points
    solutions = sorted(solutions, key=lambda x: x[0])
    pareto_front = []
    for sol in solutions:
        if not any(other[0] <= sol[0] and other[1] <= sol[1] for other in pareto_front):
            pareto_front.append(sol)
    
    return pareto_front

# Run GRASP-based Optimization with More Points
pareto_solutions = grasp_optimization(machines_df, instances_df, iterations=300)

# Plot Pareto Frontier
if pareto_solutions:
    pareto_solutions = np.array(pareto_solutions)
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_solutions[:, 0], pareto_solutions[:, 1], color='red', label='Pareto Optimal Points')
    plt.xlabel('Total Cost (CPU + RAM)')
    plt.ylabel('Load Balance (Lower is Better)')
    plt.title('Pareto Frontier: Cost vs. Load Balancing')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("No feasible Pareto solutions found.")
