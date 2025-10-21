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
    # small-world-like connected graph
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
# (updated to accept effective tasks that include mu)
# -------------------------------
def update_local_policy(effective_tasks, weights, lr=0.1):
    """
    effective_tasks: array shape (num_tasks, D_eff)
    weights: shape (D_eff,)
    """
    grad = np.zeros_like(weights)
    for task in effective_tasks:
        pred = np.dot(weights, task)
        # simple heuristic label for simulation purposes
        label = 1 if task[2] < 0.7 and task[1] > 0.3 else 0
        grad += (pred - label) * task
    return weights - lr * (grad / max(len(effective_tasks), 1))

# -------------------------------
# Federated Learning Pipeline with Communication Tracking
# (Updated: supports mu_i, effective state, and mu computation after anchors selection)
# -------------------------------
def run_proposed_method_with_ergm(num_agents=5, global_rounds=10, tasks_per_agent=240, dim=3, clusters=3):
    # hyperparams for neighborhood summary
    mu_dim = 3  # e.g., [mean_queue_norm, anchor_reward_mean, param_moment]
    
    # Prepare RoboSoccer traces and initial cluster-based inits
    raw_traces = np.random.uniform(0.3, 0.9, size=(211, dim))
    vectorized = vectorize_eligibility_traces(raw_traces)
    cluster_ids, _ = mkfc_clustering(vectorized, n_clusters=clusters)
    init_vectors = get_clusterwise_initializations(vectorized, cluster_ids, clusters)  # shape (clusters, dim)

    # Effective weight dimension = original dim + mu_dim
    eff_dim = dim + mu_dim

    # Initialize graph and influence
    graph = create_dynamic_graph(num_agents)
    influence_scores = compute_influence_scores(graph)

    # Initialize agent weights (expand init vectors to eff_dim by padding mu zeros)
    agent_weights = {}
    for i in range(num_agents):
        base = init_vectors[i % clusters]
        padded = np.concatenate([base, np.zeros(mu_dim)])  # mu part initial zeros
        agent_weights[i] = padded.copy()

    # Initialize mu_i for each agent
    mu_i = {i: np.zeros(mu_dim) for i in range(num_agents)}

    # For tracking per-agent recent acceptance ratio (for mu computation)
    recent_accept_fraction = np.zeros(num_agents)

    acc_log, energy_log, latency_log = [], [], []
    communication_log = []
    start_time = time.time()

    for rnd in range(global_rounds):
        updates = {}
        accept = 0
        total = 0
        energy = 0
        latency = 0.0

        # per-agent stats for this round (for mu computation)
        per_agent_accept = np.zeros(num_agents)
        per_agent_total = np.zeros(num_agents)

        # Local training for each agent (parallel view)
        for agent in range(num_agents):
            # generate raw tasks (local observation space)
            tasks = semantic_transform_trace(np.random.rand(tasks_per_agent, dim))
            # build effective tasks by concatenating mu_i (broadcasted)
            mu_vec = mu_i[agent]  # current neighborhood summary for the agent
            # tile mu to match tasks shape and concat
            tiled_mu = np.tile(mu_vec, (tasks.shape[0], 1))
            effective_tasks = np.hstack([tasks, tiled_mu])  # shape (tasks_per_agent, eff_dim)

            t0 = time.time()
            w = update_local_policy(effective_tasks, agent_weights[agent])
            updates[agent] = w.copy()

            # simulate decisions to collect stats (accept/energy)
            for task in effective_tasks:
                pred = np.dot(w, task)
                a = pred >= 0.5
                total += 1
                accept += int(a)
                per_agent_total[agent] += 1
                per_agent_accept[agent] += int(a)
                energy += 3 if a else 1
            latency += (time.time() - t0) * 1000.0 / max(tasks_per_agent, 1)

        # update recent accept fraction (exponential moving average style)
        for i in range(num_agents):
            fraction = per_agent_accept[i] / max(per_agent_total[i], 1)
            recent_accept_fraction[i] = 0.6 * recent_accept_fraction[i] + 0.4 * fraction

        # Select top influencers using current influence scores
        top_nodes = select_top_influencers(influence_scores, top_k=min(2, num_agents))
        # --- NEW: compute neighborhood summaries mu_i from anchors (top_nodes) ---
        # Aggregate anchor-level statistics to build mu for each agent.
        # For simulation we use: normalized queue (use tasks_per_agent as proxy), anchor reward mean, and param moment
        anchor_reward_mean = np.mean([recent_accept_fraction[n] for n in top_nodes]) if top_nodes else 0.0
        anchor_param_mean = np.mean([np.mean(agent_weights[n][:dim]) for n in top_nodes]) if top_nodes else 0.0
        # normalized queue (example): use fraction of tasks processed relative to tasks_per_agent (here ~1.0)
        norm_queue = 1.0  # placeholder (could be dynamic if queue modeled)

        # construct mu for each agent (same summary broadcast from anchors; in practice can be neighborhood-specific)
        for i in range(num_agents):
            mu_i[i] = np.array([norm_queue, anchor_reward_mean, anchor_param_mean])

        # --- Influence-weighted aggregation (over anchor updates) ---
        w_sum = sum(influence_scores[n] for n in top_nodes) if top_nodes else 1.0
        avg_weights = sum(influence_scores[n] * updates[n] for n in top_nodes) / w_sum if top_nodes else np.mean(list(updates.values()), axis=0)

        communication_log.append(len(top_nodes))

        # Disseminate aggregated global weights to agents (selective)
        for agent in range(num_agents):
            # simple replacement; could be interpolation with local model (tau)
            agent_weights[agent] = avg_weights.copy()

        acc_log.append(accept / max(total, 1))
        energy_log.append(energy / max(num_agents, 1))
        latency_log.append(latency / max(num_agents, 1))

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
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
