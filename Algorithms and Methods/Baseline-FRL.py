import numpy as np
import random
import time

# -------------------------------
# Lightweight Linear Model
# -------------------------------
class LightweightLinearModel:
    def __init__(self, input_dim=10):
        self.weights = np.random.rand(input_dim)

    def predict(self, x):
        return np.dot(self.weights, x)

    def update(self, x, reward, lr=0.05):
        prediction = self.predict(x)
        error = prediction - reward
        self.weights -= lr * error * x

# -------------------------------
# Fog Node
# -------------------------------
class FogNode:
    def __init__(self, node_id, input_dim=10):
        self.id = node_id
        self.model = LightweightLinearModel(input_dim=input_dim)
        self.input_dim = input_dim

    def train(self, episodes=100):
        for _ in range(episodes):
            state = np.random.rand(self.input_dim)
            reward = self.simulate_reward(state)
            self.model.update(state, reward)

    def simulate_reward(self, state):
        return 1 if state[2] < 0.7 and state[1] > 0.3 else 0

    def send_weights(self):
        return self.model.weights.copy()

    def receive_weights(self, weights):
        self.model.weights = weights.copy()

# -------------------------------
# Central Server
# -------------------------------
class CentralServer:
    def __init__(self, input_dim=10):
        self.global_model = LightweightLinearModel(input_dim=input_dim)
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def aggregate(self):
        all_weights = [node.send_weights() for node in self.nodes]
        avg_weights = np.mean(all_weights, axis=0)
        self.global_model.weights = avg_weights
        return len(all_weights)  # communication cost: number of uploads

    def distribute(self):
        for node in self.nodes:
            node.receive_weights(self.global_model.weights)
        return len(self.nodes)  # communication cost: number of downloads

# -------------------------------
# Federated Learning Loop with Communication Overhead Tracking
# -------------------------------
def federated_learning_loop(server, max_rounds=100, target_accuracy=0.95, tasks_per_node=240):
    accuracy_log, energy_log, latency_log = [], [], []
    comm_overhead = 0
    start_time = time.time()

    for rnd in range(max_rounds):
        print(f"\nRound {rnd + 1}/{max_rounds}")
        start_round = time.time()
        total_accepts, total_tasks, energy_used = 0, 0, 0

        for node in server.nodes:
            node.train(episodes=tasks_per_node)

        comm_overhead += server.aggregate()
        comm_overhead += server.distribute()

        for node in server.nodes:
            for _ in range(tasks_per_node):
                task = np.random.rand(node.input_dim)
                prediction = node.model.predict(task)
                accept = prediction >= 0.5
                total_tasks += 1
                total_accepts += int(accept)
                energy_used += 3 if accept else 1

        accuracy = total_accepts / total_tasks
        latency = (time.time() - start_round) * 1000 / total_tasks

        accuracy_log.append(accuracy)
        energy_log.append(energy_used / len(server.nodes))
        latency_log.append(latency)

        print(f"Accuracy: {accuracy:.4f} | ⚡ Energy: {energy_used} | 🕒 Latency: {latency:.2f} ms/task")

        if accuracy >= target_accuracy:
            print("Target accuracy reached.")
            break

    total_time = time.time() - start_time
    return {
        "avg_accuracy": np.mean(accuracy_log),
        "avg_energy": np.mean(energy_log),
        "avg_latency_ms": np.mean(latency_log),
        "convergence_rounds": rnd + 1,
        "total_simulation_time_sec": total_time,
        "total_communication_overhead": comm_overhead,
        "avg_communication_per_round": comm_overhead / (rnd + 1)
    }

# -------------------------------
# Main Function
# -------------------------------
def run_baseline_frl():
    server = CentralServer(input_dim=10)
    for i in range(5):  # 5 fog nodes
        server.add_node(FogNode(node_id=i, input_dim=10))

    results = federated_learning_loop(server)
    print("\nFinal Baseline-FRL Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_baseline_frl()
