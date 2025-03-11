# README - Multi-Objective Optimization for Fog Computing

## Introduction
This project implements a **Multi-Objective Optimization (MOO)** approach for **Load-Aware IoT Service Provisioning in Fog Computing** using a **GRASP-based metaheuristic**. The goal is to optimize the allocation of IoT services while balancing computational cost and load distribution.

## Features
- **Fog computing resource modeling** using real-world datasets.
- **GRASP-based metaheuristic** for optimization.
- **Simulated Annealing** for better exploration of solution space.
- **Parallelized execution** for faster computation.
- **Pareto Frontier visualization** for trade-off analysis between cost and load balancing.

## Prerequisites
### Required Dependencies
Ensure you have the following Python packages installed:
```bash
pip install pandas numpy networkx matplotlib joblib
```

### Required Dataset
The dataset used in this project comes from the **Chameleon Cloud Trace Dataset**, available at:  
[Chameleon CHI@UC Cloud Trace (2020-09-03)](https://www.scienceclouds.org/cloud-traces/chameleon-chiuc-cloud-trace-2020-09-03)

Place the datasets in a `data/` directory:
- `uc_machine_events.csv` (Machine event data)
- `uc_instance_events.csv` (Instance lifecycle data)

## How to Run
### Step 1: Clone the Repository
```bash
git clone https://github.com/An0lys/MOO.git
cd MOO
```

### Step 2: Run the Script
```bash
python main.py
```

### Step 3: View Results
- The script **loads machine and instance event data**.
- It **executes GRASP optimization with Simulated Annealing**.
- A **Pareto frontier plot** is displayed to visualize the trade-offs between cost and load balancing.

## Understanding the Output
- **Total Cost (CPU + RAM)**: Represents the total computational expense.
- **Load Balance (Lower is Better)**: Measures resource distribution fairness.
- **Pareto Frontier Plot**: Displays the optimal trade-off points.

## Customization
- Modify the **number of iterations** (`grasp_optimization(iterations=100)`) for more solutions.
- Adjust **Simulated Annealing parameters** (cooling rate, initial temperature).
- Use different **resource allocation strategies** by modifying `grasp_iteration()`.

## License
This project is open-source and available for academic and research purposes.