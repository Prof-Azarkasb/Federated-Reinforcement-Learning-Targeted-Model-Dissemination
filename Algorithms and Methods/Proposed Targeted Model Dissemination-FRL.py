import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
import time

# -------------------------------
# MKFC - Multi-Kernel Fuzzy Clustering
# -------------------------------
def compute_multi_kernel_matrix(X, gamma=0.5, degree=3):
    return (rbf_kernel(X, gamma=gamma) +
            linear_kernel(X) +
            polynomial_kernel(X, degree=degree)) / 3

def mkfc_clustering(X, n_clusters=3, max_iter=100, m=2.0, epsilon=1e-5):
    N = X.shape[0]
    K = compute_multi_kernel_matrix(X)
    U = np.random.dirichlet(np.ones(n_clusters), size=N)

    for _ in range(max_iter):
        U_old = U.copy()
        centers = np.array([
            np.sum((U[:, k][:, None] ** m) * K, axis=0) / np.sum(U[:, k] ** m)
            for k in range(n_clusters)
        ])
        D = np.array([
            [np.linalg.norm(K[i] - centers[k]) for k in range(n_clusters)]
            for i in range(N)
        ])
        for i in range(N):
            for k in range(n_clusters):
                denom = sum((D[i, k] / D[i, j]) ** (2 / (m - 1)) for j in range(n_clusters))
                U[i, k] = 1.0 / denom
        if np.linalg.norm(U - U_old) < epsilon:
            break
    return np.argmax(U, axis=1), U

# -------------------------------
# Semantic Alignment of Eligibility Traces
# -------------------------------
def semantic_transform_trace(trace, mapping_vector=np.array([0.85, 1.15, 1.05])):
    return trace * mapping_vector

def normalize_trace(trace):
    norm = np.linalg.norm(trace)
    return trace / norm if norm else trace

def vectorize_eligibility_traces(raw_traces):
    return np.array([semantic_transform_trace(normalize_trace(t)) for t in raw_traces])

def get_clusterwise_initializations(traces, labels, clusters):
    return np.array([np.mean(traces[labels == c], axis=0) for c in range(clusters)])

# -------------------------------
# ERGM-Based Influence Estimation
# -------------------------------
def create_dynamic_graph(n):
    return nx.connected_watts_strogatz_graph(n, k=4, p=0.3, seed=42)

def compute_influence_scores(graph, alpha=0.4, beta=0.4, gamma=0.2):
    deg = nx.degree_centrality(graph)
    btw = nx.betweenness_centrality(graph)
    clo = nx.closeness_centrality(graph)
    return {n: alpha*deg[n] + beta*btw[n] + gamma*clo[n] for n in graph.nodes}

def select_top_influencers(scores, top_k=2):
    return sorted(scores, key=scores.get, reverse=True)[:top_k]

# -------------------------------
# Local Policy Update Function
# -------------------------------
def update_local_policy(tasks, weights, lr=0.1):
    grad = np.zeros_like(weights)
    for task in tasks:
        pred = np.dot(weights, task)
        label = 1 if task[2] < 0.7 and task[1] > 0.3 else 0
        grad += (pred - label) * task
    return weights - lr * (grad / len(tasks))

# -------------------------------
# Federated Learning Pipeline with Communication Tracking
# -------------------------------
def run_proposed_method_with_ergm(num_agents=5, global_rounds=10, tasks_per_agent=240, dim=3, clusters=3):
    raw_traces = np.random.uniform(0.3, 0.9, size=(211, dim))
    vectorized = vectorize_eligibility_traces(raw_traces)
    cluster_ids, _ = mkfc_clustering(vectorized, n_clusters=clusters)
    init_vectors = get_clusterwise_initializations(vectorized, cluster_ids, clusters)
    
    graph = create_dynamic_graph(num_agents)
    influence_scores = compute_influence_scores(graph)
    agent_weights = {i: init_vectors[i % clusters].copy() for i in range(num_agents)}

    acc_log, energy_log, latency_log = [], [], []
    communication_log = []
    start_time = time.time()

    for _ in range(global_rounds):
        updates, accept, total, energy, latency = {}, 0, 0, 0, 0
        for agent in range(num_agents):
            tasks = semantic_transform_trace(np.random.rand(tasks_per_agent, dim))
            t0 = time.time()
            w = update_local_policy(tasks, agent_weights[agent])
            updates[agent] = w
            for task in tasks:
                pred = np.dot(w, task)
                a = pred >= 0.5
                total += 1
                accept += int(a)
                energy += 3 if a else 1
            latency += (time.time() - t0) * 1000 / tasks_per_agent

        top_nodes = select_top_influencers(influence_scores, top_k=2)
        w_sum = sum(influence_scores[n] for n in top_nodes)
        avg_weights = sum(influence_scores[n] * updates[n] for n in top_nodes) / w_sum

        communication_log.append(len(top_nodes))

        for agent in range(num_agents):
            agent_weights[agent] = avg_weights.copy()

        acc_log.append(accept / total)
        energy_log.append(energy / num_agents)
        latency_log.append(latency / num_agents)

    duration = time.time() - start_time
    total_comms = sum(communication_log)
    return {
        "avg_accuracy": np.mean(acc_log),
        "avg_energy": np.mean(energy_log),
        "avg_latency_ms": np.mean(latency_log),
        "convergence_rounds": global_rounds,
        "total_simulation_time_sec": duration,
        "total_communications": total_comms
    }

# -------------------------------
# Run the Simulation
# -------------------------------
if __name__ == "__main__":
    results = run_proposed_method_with_ergm()
    print("\nFinal Evaluation of Proposed Method with Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
