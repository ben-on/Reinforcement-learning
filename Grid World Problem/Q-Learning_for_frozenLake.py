import numpy as np
import gymnasium as gym
import random
from collections import defaultdict

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)

def q_learning(env, num_episodes, alpha=0.1, discount_factor=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def epsilon_greedy_policy(state, epsilon):
        if random.random() > epsilon:
            return np.argmax(Q[state])
        else:
            return env.action_space.sample()

    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = epsilon_greedy_policy(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            state = next_state

    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in Q.keys():
        best_action = np.argmax(Q[s])
        policy[s][best_action] = 1.0

    return policy, Q

policy, Q = q_learning(env, 1000)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(np.reshape(np.argmax(policy, axis=1), env.desc.shape))
