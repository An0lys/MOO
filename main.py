import pandas as pd
import numpy as np
import networkx as nx
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

# Création du graphe de Fog Computing
G = nx.Graph()
for _, row in machines_df.iterrows():
    G.add_node(row['HOST_NAME (PHYSICAL)'], **row['RESOURCES'])

# Fonction d'équilibrage de charge
def compute_load_balance(solution, num_machines):
    machine_usage = np.array([np.count_nonzero(solution == m) for m in range(num_machines)])
    return np.std(machine_usage)

# Implémentation GRASP + Recuit Simulé amélioré
def grasp_iteration(cpu_costs, ram_costs, num_machines, num_instances, simulated_annealing=False):
    solution = np.random.choice(num_machines, num_instances)
    best_solution = solution.copy()
    best_cost = np.sum(cpu_costs[solution]) + np.sum(ram_costs[solution])
    best_balance = compute_load_balance(solution, num_machines)
    
    temperature = 0.5 if simulated_annealing else 0
    cooling_rate = 0.99 if simulated_annealing else 1
    
    for _ in range(10):
        neighbor = solution.copy()
        idx = random.randint(0, num_instances - 1)
        neighbor[idx] = random.randint(0, num_machines - 1)
        
        cost_neighbor = np.sum(cpu_costs[neighbor]) + np.sum(ram_costs[neighbor])
        balance_neighbor = compute_load_balance(neighbor, num_machines)
        
        if cost_neighbor < best_cost or (simulated_annealing and np.exp((best_cost - cost_neighbor) / (temperature + 1e-5)) > random.uniform(0.5, 1)):
            best_solution = neighbor.copy()
            best_cost = cost_neighbor
            best_balance = balance_neighbor
        
        temperature *= cooling_rate
    
    return best_cost, best_balance

# Exécuter GRASP avec et sans Recuit Simulé
def evaluate_algorithms(machines_df, instances_df, iterations=100):
    num_machines = len(machines_df)
    num_instances = len(instances_df)
    
    cpu_costs = np.array([m['RESOURCES']['cpu'] for _, m in machines_df.iterrows()], dtype=float)
    ram_costs = np.array([m['RESOURCES']['ram'] for _, m in machines_df.iterrows()], dtype=float)
    
    grasp_solutions = Parallel(n_jobs=-1)(delayed(grasp_iteration)(cpu_costs, ram_costs, num_machines, num_instances, False) for _ in range(iterations))
    grasp_sa_solutions = Parallel(n_jobs=-1)(delayed(grasp_iteration)(cpu_costs, ram_costs, num_machines, num_instances, True) for _ in range(iterations))
    
    return np.array(grasp_solutions), np.array(grasp_sa_solutions)

# Comparaison des performances
grasp_solutions, grasp_sa_solutions = evaluate_algorithms(machines_df, instances_df, iterations=300)

# Filtrer les solutions dominées
def pareto_frontier(solutions):
    solutions = sorted(solutions, key=lambda x: x[0])  # Trier par coût croissant
    pareto = []
    for sol in solutions:
        if not any(other[1] <= sol[1] for other in pareto):
            pareto.append(sol)
    return np.array(pareto)

grasp_pareto = pareto_frontier(grasp_solutions)
grasp_sa_pareto = pareto_frontier(grasp_sa_solutions)

# Visualisation des Pareto Frontiers
plt.figure(figsize=(8, 6))
plt.scatter(grasp_pareto[:, 0], grasp_pareto[:, 1], color='blue', label='GRASP')
plt.scatter(grasp_sa_pareto[:, 0], grasp_sa_pareto[:, 1], color='red', label='GRASP + SA')
plt.xlabel('Total Cost (CPU + RAM)')
plt.ylabel('Load Balance (Lower is Better)')
plt.title('Comparison: GRASP vs GRASP + Simulated Annealing')
plt.legend()
plt.grid()
plt.show()