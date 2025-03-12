import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

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

# Filtrer les solutions dominées
def pareto_frontier(solutions):
    solutions = sorted(solutions, key=lambda x: x[0])  # Trier par coût croissant
    pareto = []
    for sol in solutions:
        if not any(other[1] <= sol[1] for other in pareto):
            pareto.append(sol)
    return np.array(pareto)

# Implémentation de GRASP
def grasp(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations=10):
    history = []
    solutions = []
    solution = np.random.choice(num_machines, num_instances)
    
    for iteration in range(iterations):
        neighbor = solution.copy()
        idx = random.randint(0, num_instances - 1)
        num_modifications = random.randint(1, int(0.1*len(solution)))
        for _ in range(num_modifications):
            idx = random.randint(0, num_instances - 1)
            neighbor[idx] = random.randint(0, num_machines - 1)
        
        cost_neighbor = np.sum(cpu_costs[neighbor]) + np.sum(ram_costs[neighbor]) + np.sum(storage_costs[neighbor])
        balance_neighbor = compute_load_balance(neighbor, num_machines)
        
        solutions.append((cost_neighbor, balance_neighbor))
        history.append((iteration, cost_neighbor, balance_neighbor))
        print(f"GRASP Iteration {iteration}: Cost={cost_neighbor}, Balance={balance_neighbor}")
    
    return history, np.array(solutions)

# Implémentation de l'algorithme Dragonfly
def dragonfly(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations=10):
    population_size = 30
    step_size = 0.1
    inertia_weight = 0.9
    attraction_weight = 0.5
    separation_weight = 0.2
    alignment_weight = 0.2
    cohesion_weight = 0.2
    distraction_weight = 0.1
    
    # Initialize dragonflies (positions) and velocity vectors
    population = np.random.randint(0, num_machines, (population_size, num_instances))
    velocities = np.zeros((population_size, num_instances))
    
    history = []
    solutions = []
    
    for iteration in range(iterations):
        fitness = []
        for i in range(population_size):
            solution = np.clip(np.round(population[i]), 0, num_machines - 1).astype(int)
            total_cost = np.sum(cpu_costs[solution]) + np.sum(ram_costs[solution]) + np.sum(storage_costs[solution])
            balance = compute_load_balance(solution, num_machines)
            fitness.append((total_cost, balance))
        
        best_idx = np.argmin([f[0] for f in fitness])  # Select best dragonfly based on cost
        worst_idx = np.argmax([f[0] for f in fitness])  # Select worst dragonfly
        food_source = population[best_idx]  # Best solution as food source
        enemy_source = population[worst_idx]  # Worst solution as distraction
        
        for i in range(population_size):
            neighbors = [j for j in range(population_size) if np.linalg.norm(population[j] - population[i]) < num_instances / 4]
            
            if len(neighbors) >= 1:
                separation = np.mean([population[i] - population[j] for j in neighbors], axis=0)
                alignment = np.mean([velocities[j] for j in neighbors], axis=0)
                cohesion = np.mean([population[j] for j in neighbors], axis=0) - population[i]
                attraction = food_source - population[i]
                distraction = enemy_source + population[i]
                
                velocities[i] = (inertia_weight * velocities[i] +
                                 separation_weight * separation +
                                 alignment_weight * alignment +
                                 cohesion_weight * cohesion +
                                 attraction_weight * attraction -
                                 distraction_weight * distraction)
                population[i] = np.clip(population[i] + step_size * velocities[i], 0, num_machines - 1).astype(int)
            else:
                population[i] = np.clip(population[i] + np.random.uniform(-1, 1, num_instances), 0, num_machines - 1).astype(int)
        
        solutions.append((fitness[best_idx][0], fitness[best_idx][1]))
        history.append((iteration, fitness[best_idx][0], fitness[best_idx][1]))
        print(f"Dragonfly Iteration {iteration}: Cost={fitness[best_idx][0]}, Balance={fitness[best_idx][1]}")
    
    return history, np.array(solutions)

# Exécuter GRASP et Dragonfly
def evaluate_algorithms(iterations):
    grasp_results = grasp(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations)
    dragonfly_results = dragonfly(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations)
    return grasp_results, dragonfly_results

# Comparaison des performances
grasp_results, dragonfly_results = evaluate_algorithms(iterations=30)
grasp_pareto = pareto_frontier(grasp_results[1])
dragonfly_pareto = pareto_frontier(dragonfly_results[1])

# Visualisation des Pareto Frontiers
plt.figure(figsize=(8, 6))
plt.scatter(grasp_results[1][:, 0], grasp_results[1][:, 1], color='lightblue', alpha=0.5, label='GRASP (all)')
plt.scatter(dragonfly_results[1][:, 0], dragonfly_results[1][:, 1], color='lightgreen', alpha=0.5, label='Dragonfly (all)')
plt.scatter(grasp_pareto[:, 0], grasp_pareto[:, 1], color='blue', label='GRASP Pareto')
plt.scatter(dragonfly_pareto[:, 0], dragonfly_pareto[:, 1], color='green', label='Dragonfly Pareto')
plt.xlabel('Total Cost (CPU + RAM + Storage)')
plt.ylabel('Load Balance (Lower is Better)')
plt.title('Comparison: GRASP vs Dragonfly')
plt.legend()
plt.grid()
plt.show()
