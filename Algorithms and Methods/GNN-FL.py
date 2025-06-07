import numpy as np
import networkx as nx
import time

# Generate random tasks: [complexity, deadline, resource need]
def generate_tasks(num_tasks):
    return np.random.rand(num_tasks, 3)

# Create synthetic communication graph among fog nodes
def create_topology(num_nodes):
    G = nx.connected_watts_strogatz_graph(n=num_nodes, k=3, p=0.3, seed=42)
    return G

# Message aggregation based on neighbors in GNN-style update
def aggregate_messages(graph, node_features):
    aggregated = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            aggregated[node] = node_features[node]
        else:
            neighbor_features = np.mean([node_features[n] for n in neighbors], axis=0)
            aggregated[node] = (node_features[node] + neighbor_features) / 2
    return aggregated

# Local training with GNN-like update
def local_gnn_training(tasks, weights, learning_rate=0.1):
    gradients = np.zeros_like(weights)
    for task in tasks:
        prediction = np.dot(weights, task)
        label = 1 if task[2] < 0.7 and task[1] > 0.3 else 0
        error = prediction - label
        gradients += error * task
    gradients /= len(tasks)
    new_weights = weights - learning_rate * gradients
    return new_weights

# GNN-FL Evaluation
def evaluate_gnn_fl(num_nodes=5, global_rounds=10, tasks_per_node=240):
    dim = 3
    graph = create_topology(num_nodes)
    node_weights = {i: np.random.rand(dim) for i in range(num_nodes)}
    accuracy_log, energy_log, latency_log = [], [], []
    start_time = time.time()

    for rnd in range(global_rounds):
        node_features = {}
        accept_total, task_total, energy_total, latency_total = 0, 0, 0, 0

        for node in range(num_nodes):
            tasks = generate_tasks(tasks_per_node)
            start_node_time = time.time()

            updated_weights = local_gnn_training(tasks, node_weights[node])
            node_features[node] = updated_weights

            for task in tasks:
                pred = np.dot(updated_weights, task)
                accept = 1 if pred >= 0.5 else 0
                task_total += 1
                if accept:
                    accept_total += 1
                    energy_total += 5
                else:
                    energy_total += 1

            latency_total += (time.time() - start_node_time) * 1000 / tasks_per_node

        # Message passing
        node_weights = aggregate_messages(graph, node_features)

        acc = accept_total / task_total
        accuracy_log.append(acc)
        energy_log.append(energy_total / num_nodes)
        latency_log.append(latency_total / num_nodes)

    duration = time.time() - start_time

    return {
        "avg_accuracy": np.mean(accuracy_log),
        "avg_energy": np.mean(energy_log),
        "avg_latency_ms": np.mean(latency_log),
        "convergence_rounds": global_rounds,
        "total_time_sec": duration
    }

# Run evaluation
results = evaluate_gnn_fl()
print("\nGNN-FL Evaluation Results:")
for key, val in results.items():
    print(f"{key}: {val:.4f}")
