import numpy as np
import time

# Simulated Task Generation
def generate_tasks(num_tasks):
    # Each task: [complexity, deadline, resource_requirement] ∈ [0,1]
    return np.random.rand(num_tasks, 3)

# Local training at each fog node
def local_training(tasks, weights, learning_rate=0.1):
    gradients = np.zeros_like(weights)
    for task in tasks:
        prediction = np.dot(weights, task)
        label = 1 if task[2] < 0.7 and task[1] > 0.3 else 0  # acceptance criteria
        error = prediction - label
        gradients += error * task
    gradients /= len(tasks)
    new_weights = weights - learning_rate * gradients
    return new_weights

# Federated Averaging function
def fed_avg(weights_list):
    return np.mean(weights_list, axis=0)

# Main Evaluation Function
def evaluate_fedavg(num_clients=5, global_rounds=10, tasks_per_client=240):
    dim = 3  # number of task features
    global_weights = np.random.rand(dim)
    accuracy_log = []
    energy_log = []
    latency_log = []
    start_time = time.time()

    for rnd in range(global_rounds):
        local_weights_list = []
        accept_total, task_total, energy_total, latency_total = 0, 0, 0, 0

        for c in range(num_clients):
            tasks = generate_tasks(tasks_per_client)
            local_weights = np.copy(global_weights)
            start_node_time = time.time()

            # Local model training at each client
            local_weights = local_training(tasks, local_weights)
            local_weights_list.append(local_weights)

            # Evaluation after local training
            for task in tasks:
                pred = np.dot(local_weights, task)
                accept = 1 if pred >= 0.5 else 0
                task_total += 1
                if accept == 1:
                    accept_total += 1
                    energy_total += 5  # accept = 5 Joules
                else:
                    energy_total += 1  # reject = 1 Joule

            latency_total += (time.time() - start_node_time) * 1000 / tasks_per_client  # ms per task

        # Aggregation step (FedAvg)
        global_weights = fed_avg(local_weights_list)

        # Performance logging
        accuracy = accept_total / task_total
        avg_energy = energy_total / num_clients
        avg_latency = latency_total / num_clients

        accuracy_log.append(accuracy)
        energy_log.append(avg_energy)
        latency_log.append(avg_latency)

    duration = time.time() - start_time

    return {
        "avg_accuracy": np.mean(accuracy_log),
        "avg_energy": np.mean(energy_log),
        "avg_latency_ms": np.mean(latency_log),
        "convergence_rounds": global_rounds,
        "total_time_sec": duration
    }

# Run the evaluation
results = evaluate_fedavg()
print("\nFedAvg Evaluation Results:")
for key, val in results.items():
    print(f"{key}: {val:.4f}")
