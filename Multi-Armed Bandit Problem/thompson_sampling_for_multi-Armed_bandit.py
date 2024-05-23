import numpy as np

class ThompsonSampling:
    def __init__(self, k):
        self.k = k
        self.successes = np.zeros(k)
        self.failures = np.zeros(k)

    def select_action(self):
        beta_samples = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1) for i in range(self.k)]
        return np.argmax(beta_samples)

    def update(self, action, reward):
        if reward == 1:
            self.successes[action] += 1
        else:
            self.failures[action] += 1

def simulate_bandit(agent, bandit_probs, num_steps):
    rewards = np.zeros(num_steps)
    for step in range(num_steps):
        action = agent.select_action()
        reward = np.random.binomial(1, bandit_probs[action])
        agent.update(action, reward)
        rewards[step] = reward
    return rewards

k = 10
bandit_probs = np.random.rand(k)
num_steps = 1000

thompson_sampling_agent = ThompsonSampling(k)
thompson_sampling_rewards = simulate_bandit(thompson_sampling_agent, bandit_probs, num_steps)
print(f"Thompson Sampling Total Reward: {np.sum(thompson_sampling_rewards)}")
