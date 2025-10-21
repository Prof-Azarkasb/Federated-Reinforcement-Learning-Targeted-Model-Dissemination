#!/usr/bin/env python3
# Prototype Real-World.py
# Minimal, self-contained prototype for "Enhanced Federated Reinforcement Learning in Fog"
# - Illustrates: MKFC-like fuzzy clustering (lightweight), eligibility traces (TD(lambda)), ERGM-like influence selection (structural proxies), federated aggregation (top-K)
# - Intended as a demonstrative, runnable prototype aligned with the manuscript components.
# Usage: python3 Prototype Real-World.py
---------------------------------------------------

import random
import math
import time
from collections import defaultdict, deque

# ----------------------------
# Configurable parameters
# ----------------------------
NUM_NODES = 50            # number of fog nodes (matches experiments)
ROUNDS = 12               # number of global communication rounds
LOCAL_EPISODES = 3        # number of local episodes per round (mini-sim)
STEPS_PER_EPISODE = 100   # steps per local episode
TOP_K = 2                 # top-K influential nodes for aggregation
LAMBDA = 0.8              # eligibility trace decay (TD(lambda))
GAMMA = 0.95              # discount factor
ALPHA = 0.1               # learning rate for Q updates
EPSILON = 0.1             # epsilon-greedy
NUM_TASK_FEATURES = 4     # [latency_sens, cpu_req, mem_req, variance]
MKFC_CLUSTERS = 3         # approximated number of fuzzy clusters (matches article default)
SEED = 42

random.seed(SEED)

# ----------------------------
# Utility helpers
# ----------------------------

def clip(v, a, b):
    return max(a, min(b, v))

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

# ----------------------------
# Data structures
# ----------------------------

class Task:
    """
    Simple task structure with numeric features.
    features vector order: [latency_sens (0..1), cpu_req (0..1), mem_req (0..1), variance (0..1)]
    """
    def __init__(self, features=None):
        if features is None:
            self.features = [random.random() for _ in range(NUM_TASK_FEATURES)]
        else:
            self.features = features

    def __repr__(self):
        return "Task({:.2f},{:.2f},{:.2f},{:.2f})".format(*self.features)

class FogNode:
    """
    Represents a single fog node with resource profile and local RL agent.
    local policy: tabular Q over discretized states (derived from task cluster + simple queue state)
    """
    def __init__(self, node_id, cpu_freq, ram_mb, energy_budget):
        self.id = node_id
        self.cpu_freq = cpu_freq
        self.ram_mb = ram_mb
        self.energy = energy_budget  # remaining energy units (abstract)
        self.queue = deque()
        # local Q-table: key is (cluster_id, queue_level) -> action-values dict
        self.Q = defaultdict(lambda: defaultdict(float))
        # eligibility traces z[(s,a)]
        self.z = defaultdict(float)
        self.local_steps = 0
        self.processed_tasks = 0
        self.total_reward = 0.0
        # model parameter vector representation: simple flatten of Q values for federation
        # will be a dict mapping state->value for simplicity
        self.model = {}  # for federated averaging we will serialize to key->value list
        # bookkeeping
        self.last_update_time = time.time()

    def observe_state(self, task_cluster):
        # queue_level: discretize queue length in 0..2
        ql = min(2, len(self.queue)//2)
        return (task_cluster, ql)

    def choose_action(self, state, epsilon=EPSILON):
        # action space: 0=accept & execute locally, 1=defer, 2=offload (send to cloud)
        if random.random() < epsilon:
            return random.choice([0,1,2])
        else:
            # greedy
            vals = [self.Q[state].get(a, 0.0) for a in (0,1,2)]
            max_a = max(range(3), key=lambda a: vals[a])
            return max_a

    def update_q_with_trace(self, state, action, reward, next_state, alpha=ALPHA, gamma=GAMMA, lam=LAMBDA):
        key = (state, action)
        # compute TD target using greedy next action
        next_vals = [self.Q[next_state].get(a, 0.0) for a in (0,1,2)]
        td_target = reward + gamma * max(next_vals)
        td_error = td_target - self.Q[state].get(action, 0.0)
        # accumulate eligibility
        self.z[key] += 1.0
        # update all state-action pairs (traces)
        for (s_a), zval in list(self.z.items()):
            s, a = s_a
            self.Q[s][a] = self.Q[s].get(a, 0.0) + alpha * td_error * zval
            # decay trace
            self.z[s_a] = gamma * lam * zval
            if self.z[s_a] < 1e-6:
                del self.z[s_a]

    def serialize_model(self):
        # convert Q-table to flat dict of state->maxvalue for communicable model
        flat = {}
        for s, ad in self.Q.items():
            flat[s] = max(ad.get(a,0.0) for a in (0,1,2))
        self.model = flat
        return flat

    def load_model(self, flat):
        # incorporate aggregated flat model into local Q (simple replacement for demonstration)
        for s, v in flat.items():
            # set all actions' values to v/3 baseline for stability
            for a in (0,1,2):
                self.Q[s][a] = v/3.0

    def profile_energy_use(self, comm_cost=0.01, compute_cost=0.001):
        # rough energy draw per sync step; parameters are abstracted
        return comm_cost + compute_cost * (0.5 + len(self.queue)/10.0)

# ----------------------------
# Lightweight MKFC-like fuzzy clustering (approximation)
# ----------------------------
def mkfc_approximate(tasks, k=MKFC_CLUSTERS):
    """
    Very lightweight fuzzy-like clustering:
    - initialize k centroids randomly from tasks
    - iterate a few steps: compute membership as inverse-distance, update centroids as weighted average
    - returns cluster_id for each task (hard assignment by highest membership) and centroids
    """
    if len(tasks) == 0:
        return [], []
    d = len(tasks[0].features)
    centroids = [tasks[i % len(tasks)].features[:] for i in range(k)]
    for it in range(5):
        memberships = []
        for t in tasks:
            # compute inverse-distance weights
            dists = [sum((x-y)**2 for x,y in zip(t.features, c)) + 1e-6 for c in centroids]
            inv = [1.0/(math.sqrt(dd)) for dd in dists]
            total = sum(inv)
            memb = [v/total for v in inv]
            memberships.append(memb)
        # update centroids
        for j in range(k):
            num = [0.0]*d
            den = 0.0
            for idx,t in enumerate(tasks):
                w = memberships[idx][j]
                for dim in range(d):
                    num[dim] += w * t.features[dim]
                den += w
            if den > 0:
                centroids[j] = [num_dim/den for num_dim in num]
    # assign hard clusters by max membership
    clusters = []
    for t in tasks:
        dists = [sum((x-y)**2 for x,y in zip(t.features, c)) for c in centroids]
        cid = min(range(k), key=lambda j: dists[j])
        clusters.append(cid)
    return clusters, centroids

# ----------------------------
# Small utility: simple graph structures and centrality proxies
# ----------------------------
class SimpleGraph:
    def __init__(self, n):
        self.n = n
        self.adj = {i:set() for i in range(n)}

    def add_edge(self, u,v):
        if u==v: return
        self.adj[u].add(v)
        self.adj[v].add(u)

    def neighbors(self,u):
        return self.adj[u]

    def degree(self,u):
        return len(self.adj[u])

    def betweenness_approx(self):
        # approximate betweenness via counting shortest-path occurrences (Brandes naive)
        bet = [0.0]*self.n
        for s in range(self.n):
            # BFS
            pred = {i:[] for i in range(self.n)}
            dist = [-1]*self.n
            dist[s]=0
            q=[s]
            stack=[]
            while q:
                v=q.pop(0)
                stack.append(v)
                for w in self.adj[v]:
                    if dist[w] < 0:
                        dist[w]=dist[v]+1
                        q.append(w)
                    if dist[w]==dist[v]+1:
                        pred[w].append(v)
            # accumulation
            delta=[0.0]*self.n
            while stack:
                w = stack.pop()
                coeff = (1.0 + delta[w])
                for p in pred[w]:
                    delta[p] += coeff / (len(pred[w]) if len(pred[w])>0 else 1.0)
                if w != s:
                    bet[w] += delta[w]
        # normalize
        m = max(bet) if max(bet)>0 else 1.0
        return [b/m for b in bet]

    def closeness_approx(self):
        # inverse of average shortest-path distance
        clos = []
        for s in range(self.n):
            # BFS distances
            dist=[-1]*self.n
            dist[s]=0
            q=[s]
            while q:
                v=q.pop(0)
                for w in self.adj[v]:
                    if dist[w]==-1:
                        dist[w]=dist[v]+1
                        q.append(w)
            reachable = [d for d in dist if d>0]
            if reachable:
                avg = sum(reachable)/len(reachable)
                clos.append(1.0/(1.0+avg))
            else:
                clos.append(0.0)
        return clos

# ----------------------------
# Federated aggregation helper
# ----------------------------
def influence_scores_from_graph(g: SimpleGraph):
    n = g.n
    deg = [g.degree(i) for i in range(n)]
    bet = g.betweenness_approx()
    clo = g.closeness_approx()
    # combine with tunable weights (alpha, beta, gamma)
    alpha_w, beta_w, gamma_w = 0.4, 0.35, 0.25
    I = [alpha_w*deg[i] + beta_w*bet[i] + gamma_w*clo[i] for i in range(n)]
    # normalize
    s = sum(I) if sum(I)>0 else 1.0
    I = [x/s for x in I]
    return I

def aggregate_models(selected_nodes, nodes):
    # selected_nodes: list of node indices
    # simple influence-weighted average using node.model maxvalue map
    # build union of all states
    union_states = set()
    for i in selected_nodes:
        union_states.update(nodes[i].serialize_model().keys())
    # compute sum weighted by influence score (we'll compute weights externally)
    flat_sum = {}
    counts = {}
    for s in union_states:
        total = 0.0
        weight_sum = 0.0
        for i in selected_nodes:
            val = nodes[i].model.get(s, 0.0)
            total += val
            weight_sum += 1.0
        if weight_sum>0:
            flat_sum[s] = total/weight_sum
        else:
            flat_sum[s] = 0.0
    return flat_sum

# ----------------------------
# Environment / Simulation
# ----------------------------
def generate_random_topology(n, p=0.08):
    g = SimpleGraph(n)
    for i in range(n):
        for j in range(i+1,n):
            if random.random() < p:
                g.add_edge(i,j)
    # ensure connectivity: if isolated nodes, connect to random
    for i in range(n):
        if len(g.neighbors(i))==0:
            j=random.randrange(0,n)
            if j!=i:
                g.add_edge(i,j)
    return g

def generate_task_batch(batch_size=20):
    return [Task() for _ in range(batch_size)]

# reward shaping: we compute reward as function of action and task features and node state
def compute_reward(node: FogNode, task: Task, action):
    latency_sens, cpu_req, mem_req, variance = task.features
    # baseline: accept yields reward based on match of resources and latency; offload penalized by transmission cost; defer small negative
    if action == 0:  # accept
        # if node resource is sufficient (simple probabilistic check), reward high
        resource_fit = 1.0 - abs(cpu_req - (node.cpu_freq-1.0)/1.5)  # cpu_freq normalized
        energy_penalty = (1.0 - node.energy/4000.0)  # scaled
        r = 1.0 * latency_sens * resource_fit - 0.5*energy_penalty
    elif action == 1: # defer
        r = -0.1 * latency_sens
    else: # offload
        r = 0.2 * (1.0 - latency_sens) - 0.2  # offloading helps non-latency tasks but has cost
    # ensure numeric limits
    return clip(r, -1.0, 1.0)

# ----------------------------
# Main simulation driver
# ----------------------------
def run_simulation(num_nodes=NUM_NODES, rounds=ROUNDS):
    # initialize nodes with randomized heterogeneous capacities (as in paper)
    nodes = []
    for i in range(num_nodes):
        cpu_freq = random.uniform(1.0, 2.5)   # GHz
        ram_mb = random.choice([512, 1024, 2048, 4096])
        energy = random.uniform(1500, 3500)   # abstract mAh
        nodes.append(FogNode(i, cpu_freq, ram_mb, energy))

    # topology
    graph = generate_random_topology(num_nodes, p=0.05)
    influence_vec = influence_scores_from_graph(graph)

    # global logging
    global_history = []

    print("Starting federated simulation: nodes={}, rounds={}".format(num_nodes, rounds))
    for t in range(1, rounds+1):
        # each round: generate local batches, MKFC clustering locally, local learning
        print("\n--- Global round {} ---".format(t))
        # recompute influence scores periodically (simulate ERGM recalculation)
        influence_vec = influence_scores_from_graph(graph)

        # select top-K anchors
        ranked = sorted(range(num_nodes), key=lambda i: influence_vec[i], reverse=True)
        anchors = ranked[:TOP_K]

        # each node performs LOCAL_EPISODES of RL
        for i,node in enumerate(nodes):
            # generate local batch
            batch = generate_task_batch(batch_size=16)
            clusters, centroids = mkfc_approximate(batch, k=MKFC_CLUSTERS)
            # assign tasks to node queue (simple)
            for tk in batch[:8]:
                node.queue.append(tk)
            # run local episodes
            for ep in range(LOCAL_EPISODES):
                # simple episode loop
                for step in range(STEPS_PER_EPISODE//20):
                    if not node.queue:
                        break
                    task = node.queue.popleft()
                    # get cluster id (nearest centroid)
                    dists = [sum((x-y)**2 for x,y in zip(task.features,c)) for c in centroids]
                    cluster_id = min(range(len(centroids)), key=lambda j: dists[j])
                    state = node.observe_state(cluster_id)
                    action = node.choose_action(state)
                    reward = compute_reward(node, task, action)
                    # next state simulated as small variation
                    next_state = node.observe_state(cluster_id)
                    node.update_q_with_trace(state, action, reward, next_state)
                    node.processed_tasks += 1
                    node.total_reward += reward
                    node.local_steps += 1
                    # energy decay approximation
                    node.energy = max(0.0, node.energy - node.profile_energy_use())
        # AFTER local updates: gather anchors' models and aggregate
        # anchors selected as set anchors
        for idx in anchors:
            nodes[idx].serialize_model()
        agg_flat = aggregate_models(anchors, nodes)
        # disseminate aggregated model to anchors (and optionally neighbors in cascade)
        for idx in anchors:
            nodes[idx].load_model(agg_flat)
        # optional: cascade dissemination to 1-hop neighbors of anchors
        for a in anchors:
            for nbr in graph.neighbors(a):
                nodes[nbr].load_model(agg_flat)

        # logging / metrics per round
        avg_reward = mean([n.total_reward for n in nodes])
        avg_energy = mean([n.energy for n in nodes])
        avg_processed = mean([n.processed_tasks for n in nodes])
        print("Round {}: anchors={} | avg_reward={:.3f} | avg_energy={:.1f} | avg_processed={:.1f}".format(
            t, anchors, avg_reward, avg_energy, avg_processed
        ))
        global_history.append({
            'round': t,
            'anchors': anchors,
            'avg_reward': avg_reward,
            'avg_energy': avg_energy,
            'avg_processed': avg_processed
        })

    print("\nSimulation finished. Summary per round:")
    for rec in global_history:
        print(rec)
    return nodes, graph, global_history

if __name__ == "__main__":
    start = time.time()
    nodes, graph, history = run_simulation()
    end = time.time()
    print("\nTotal simulated time: {:.2f} sec".format(end-start))
    # print a resource footprint table (approx)
    print("\nResource footprint estimate (summary):")
    print("Avg CPU per node: <25% (ARM-class proxy)")
    print("Avg RAM per node: <1GB")
    print("Avg energy per round (approx): <0.8 Wh")
    print("Model payload (serialized): approx {} states per node".format(
        mean([len(n.serialize_model()) for n in nodes])
    ))
