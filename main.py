import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed

# Charger les fichiers CSV
machine_events_path = "data/uc_machine_events.csv"
instance_events_path = "data/uc_instance_events.csv"

machines_df = pd.read_csv(machine_events_path)
instances_df = pd.read_csv(instance_events_path)

# Extraction des ressources
def parse_properties(properties_str):
    try:
        return eval(properties_str)  # Convertir string en dictionnaire
    except:
        return {}

machines_df['PROPERTIES'] = machines_df['PROPERTIES'].apply(parse_properties)

def extract_resources(row):
    props = row['PROPERTIES']
    return {
        'cpu': float(props.get('architecture.smt_size', 0)),
        'ram': float(props.get('main_memory.ram_size', 0)),
        'storage': float(props.get('storage_device.size', 0))
    }

machines_df['RESOURCES'] = machines_df.apply(extract_resources, axis=1)

# Récupération des coûts depuis les fichiers CSV
num_machines = len(machines_df)
num_instances = len(instances_df)

cpu_costs = np.array([m['RESOURCES']['cpu'] for _, m in machines_df.iterrows()], dtype=float)
ram_costs = np.array([m['RESOURCES']['ram'] for _, m in machines_df.iterrows()], dtype=float)
storage_costs = np.array([m['RESOURCES']['storage'] for _, m in machines_df.iterrows()], dtype=float)

# Fonction d'équilibrage de charge
def compute_load_balance(solution, num_machines):
    machine_usage = np.array([np.count_nonzero(solution == m) for m in range(num_machines)])
    return np.std(machine_usage)

# Implémentation de GRASP
def grasp_iteration(cpu_costs, ram_costs, storage_costs, num_machines, num_instances):
    solution = np.random.choice(num_machines, num_instances)
    best_solution = solution.copy()
    best_cost = np.sum(cpu_costs[solution]) + np.sum(ram_costs[solution]) + np.sum(storage_costs[solution])
    best_balance = compute_load_balance(solution, num_machines)
    
    for _ in range(10):
        neighbor = solution.copy()
        idx = random.randint(0, num_instances - 1)
        neighbor[idx] = random.randint(0, num_machines - 1)
        
        cost_neighbor = np.sum(cpu_costs[neighbor]) + np.sum(ram_costs[neighbor]) + np.sum(storage_costs[neighbor])
        balance_neighbor = compute_load_balance(neighbor, num_machines)
        
        if cost_neighbor < best_cost:
            best_solution = neighbor.copy()
            best_cost = cost_neighbor
            best_balance = balance_neighbor
    

    print(f"Cost={best_cost}, Balance={best_balance}")
    return best_cost, best_balance

# Implémentation de l'algorithme Dragonfly
def dragonfly_optimization(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations=10):
    population_size = 30
    step_size = 0.1
    inertia_weight = 0.9
    attraction_weight = 0.5
    
    population = np.random.randint(0, num_machines, (population_size, num_instances))
    velocities = np.random.uniform(-1, 1, (population_size, num_instances))
    
    best_solution = None
    best_cost = float('inf')
    best_balance = float('inf')
    
    for _ in range(iterations):
        for i in range(population_size):
            solution = np.clip(np.round(population[i]), 0, num_machines - 1).astype(int)
            total_cost = np.sum(cpu_costs[solution]) + np.sum(ram_costs[solution]) + np.sum(storage_costs[solution])
            balance = compute_load_balance(solution, num_machines)
            
            if total_cost < best_cost or (total_cost == best_cost and balance < best_balance):
                best_solution = solution.copy()
                best_cost = total_cost
                best_balance = balance
            
            velocities[i] = inertia_weight * velocities[i] + attraction_weight * (best_solution - population[i])
            population[i] = np.clip(population[i] + step_size * velocities[i], 0, num_machines - 1).astype(int)
    
    print(f"Cost={best_cost}, Balance={best_balance}")
    return best_cost, best_balance

# Exécuter GRASP et Dragonfly
def evaluate_algorithms(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations=100):
    print("Running GRASP...")
    grasp_solutions = Parallel(n_jobs=-1)(delayed(grasp_iteration)(cpu_costs, ram_costs, storage_costs, num_machines, num_instances) for _ in range(iterations))

    print("Running Dragonfly...")
    dragonfly_solutions = Parallel(n_jobs=-1)(delayed(dragonfly_optimization)(cpu_costs, ram_costs, storage_costs, num_machines, num_instances) for _ in range(iterations))
    
    return np.array(grasp_solutions), np.array(dragonfly_solutions)

# Comparaison des performances
grasp_solutions, dragonfly_solutions = evaluate_algorithms(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations=10)

def pareto_frontier(solutions):
    solutions = sorted(solutions, key=lambda x: x[0])  # Trier par coût croissant
    pareto = []
    for sol in solutions:
        if not any(other[1] <= sol[1] for other in pareto):
            pareto.append(sol)
    return np.array(pareto)

grasp_pareto = pareto_frontier(grasp_solutions)
dragonfly_pareto = pareto_frontier(dragonfly_solutions)

# Visualisation des Pareto Frontiers
plt.figure(figsize=(8, 6))
plt.scatter(grasp_solutions[:, 0], grasp_solutions[:, 1], color='lightblue', alpha=0.5, label='GRASP (all)')
plt.scatter(dragonfly_solutions[:, 0], dragonfly_solutions[:, 1], color='lightgreen', alpha=0.5, label='Dragonfly (all)')
plt.scatter(grasp_pareto[:, 0], grasp_pareto[:, 1], color='blue', label='GRASP Pareto')
plt.scatter(dragonfly_pareto[:, 0], dragonfly_pareto[:, 1], color='green', label='Dragonfly Pareto')
plt.xlabel('Total Cost (CPU + RAM + Storage)')
plt.ylabel('Load Balance (Lower is Better)')
plt.title('Comparison: GRASP vs Dragonfly')
plt.legend()
plt.grid()
plt.show()
