import numpy as np
import time

# Simulated Environment for Fog Task Scheduling
class FogEnvironment:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        self.current_task = 0
        self.tasks = self._generate_tasks(n_tasks)

    def _generate_tasks(self, n):
        # Task features: [complexity, deadline, resource_need] ∈ [0, 1]
        return np.random.rand(n, 3)

    def reset(self):
        self.current_task = 0
        return self.tasks[self.current_task]

    def step(self, action):
        task = self.tasks[self.current_task]
        done = self.current_task == self.n_tasks - 1

        # Reward logic
        reward = 1 if action == 1 and task[2] < 0.7 and task[1] > 0.3 else -1

        self.current_task += 1
        next_state = None if done else self.tasks[self.current_task]
        return next_state, reward, done

# Simple Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((10, 10, 2))  # Discrete state-action table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _discretize(self, state):
        return tuple((state[:2] * 10).astype(int))

    def choose_action(self, state):
        s = self._discretize(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[s])

    def update(self, state, action, reward, next_state):
        s = self._discretize(state)
        ns = self._discretize(next_state) if next_state is not None else s
        best_next = np.max(self.q_table[ns])
        self.q_table[s][action] += self.alpha * (reward + self.gamma * best_next - self.q_table[s][action])

# Evaluation Function
def evaluate_rl(n_episodes=30, steps=1200):
    acc_curve = []
    energy_curve = []
    latency_curve = []
    total_rewards = []

    for ep in range(n_episodes):
        env = FogEnvironment(n_tasks=steps)
        agent = QLearningAgent()
        state = env.reset()
        rewards = 0
        accept_count = 0
        energy = 0
        start_ep_time = time.time()

        for _ in range(steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            rewards += reward
            energy += 5 if action == 1 else 1  # Energy model: 5J for processing, 1J otherwise
            if action == 1:
                accept_count += 1
            if done:
                break

        acc = accept_count / steps
        latency = (time.time() - start_ep_time) * 1000 / steps  # in milliseconds
        acc_curve.append(acc)
        energy_curve.append(energy)
        latency_curve.append(latency)
        total_rewards.append(rewards)

    return {
        "avg_accuracy": np.mean(acc_curve),
        "avg_energy": np.mean(energy_curve),
        "avg_latency_ms": np.mean(latency_curve),
        "convergence_time_steps": steps,
        "average_reward": np.mean(total_rewards)
    }

# Run the evaluation
results = evaluate_rl()
print("Classical RL Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
