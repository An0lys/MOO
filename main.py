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
   
    # 1) Conversion en liste de tableaux NumPy si besoin (pour uniformiser le type et faciliter zip).
    solutions = [np.array(s, dtype=float) for s in solutions]

    # 2) On trie par la première dimension (ex: coût).
    solutions.sort(key=lambda x: x[0])
    
    pareto = []
    
    # 3) Pour chaque solution, on vérifie si elle est dominée ou non par un membre de la frontière.
    for sol in solutions:
        # Si aucune solution dans 'pareto' ne domine 'sol', alors 'sol' est Pareto-optimal.
        # 'o' domine 'sol' si o[i] <= sol[i] pour toutes les dimensions i.
        # On cherche donc s'il existe un 'o' qui vérifie cette condition.
        is_dominated = any(
            all(o_i <= s_i for o_i, s_i in zip(o, sol))
            for o in pareto
        )
        if not is_dominated:
            pareto.append(sol)

    # 4) Conversion finale en array.
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
def dragonfly(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations):
    population_size = 30
    
    inertia_weight = 0.9
    attraction_weight = 0.5
    separation_weight = 0.2
    alignment_weight = 0.2
    cohesion_weight = 0.2
    distraction_weight = 0.1
    distance_threshold = None
    balance_weight=1.0
    
    # Pré-calcul des coûts
    machine_costs = cpu_costs + ram_costs + storage_costs

    # Initialisation
    population = np.random.randint(0, num_machines, size=(population_size, num_instances))
    velocities = np.zeros_like(population, dtype=float)

    # Pour mémoriser les évolutions
    history = []
    solutions = []

    best_cost = np.inf
    
    best_balance = 0

    if distance_threshold is None:
        distance_threshold = num_instances / 4.0

    for iteration in range(int(iterations / population_size)):
        # Calcul vectorisé
        total_costs = np.array([np.sum(machine_costs[ind]) for ind in population])
        balances = np.array([compute_load_balance(ind, num_machines) for ind in population])
        fitness = total_costs - balance_weight *  balances

        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        food_source = population[best_idx].copy()
        enemy_source = population[worst_idx].copy()

        diff = population[:, None, :] - population[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1))
        neighbors_mask = (distances < distance_threshold)

        for i in range(population_size):
            neighbors_idx = np.where(neighbors_mask[i])[0]
            n_count = len(neighbors_idx)

            if n_count > 1:
                separation = np.sum(population[i] - population[neighbors_idx], axis=0) / n_count
                alignment = np.mean(velocities[neighbors_idx], axis=0)
                mean_position = np.mean(population[neighbors_idx], axis=0)
                cohesion = mean_position - population[i]
            else:
                separation = np.random.uniform(-1, 1, num_instances)
                alignment = np.zeros(num_instances)
                cohesion = np.zeros(num_instances)

            attraction = food_source - population[i]
            distraction = enemy_source - population[i]

            velocities[i] = (
                inertia_weight * velocities[i]
                + separation_weight * separation
                + alignment_weight * alignment
                + cohesion_weight * cohesion
                + attraction_weight * attraction
                - distraction_weight * distraction
            )

            population[i] = np.round(population[i] + velocities[i]).astype(int)
            np.clip(population[i], 0, num_machines - 1, out=population[i])

        # Meilleur individu global de cette itération
        iteration_best_cost = total_costs[best_idx]
        iteration_best_balance = balances[best_idx]

        # Mise à jour si on a un mieux global
        if iteration_best_balance > best_balance:
            
            
            best_balance = iteration_best_balance
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                 

        # On mémorise dans l'historique
        history.append((iteration, iteration_best_cost, iteration_best_balance))
        solutions.append((best_cost, best_balance))

        print(f"Iteration {iteration} | Best Cost: {best_cost} | Best balance: {best_balance}" )


    # On retourne, comme GRASP, l'historique et toutes les paires (coût, balance)
    return history, np.array(solutions)


# Exécuter GRASP et Dragonfly
def evaluate_algorithms(iterations):
    grasp_results = grasp(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations)
    dragonfly_results = dragonfly(cpu_costs, ram_costs, storage_costs, num_machines, num_instances, iterations)
    return grasp_results, dragonfly_results

# Comparaison des performances
grasp_results, dragonfly_results = evaluate_algorithms(iterations=900)
grasp_pareto = pareto_frontier(grasp_results[1])
dragonfly_pareto = pareto_frontier(dragonfly_results[1])

# Visualisation des Pareto Frontiers
plt.figure(figsize=(8, 6))
plt.scatter(grasp_results[1][:, 0], grasp_results[1][:, 1], color='lightblue', alpha=0.5, label='GRASP (all)')
plt.scatter(dragonfly_results[1][:, 0], dragonfly_results[1][:, 1], color='lightgreen', alpha=0.5, label='Dragonfly (all)')
plt.scatter(grasp_pareto[:, 0], grasp_pareto[:, 1], color='blue', label='GRASP Pareto')
plt.scatter(dragonfly_pareto[:, 0], dragonfly_pareto[:, 1], color='green', label='Dragonfly Pareto')
plt.xlabel('Total Cost (CPU + RAM + Storage)')
plt.ylabel('Load Balance (Higher is Better)')
plt.title('Comparison: GRASP vs Dragonfly')
plt.legend()
plt.grid()
plt.show()
