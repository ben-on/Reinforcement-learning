import numpy as np
import gymnasium as gym

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)

def value_iteration(env, discount_factor=1.0, theta=1e-9):
    def one_step_lookahead(state, V):
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy, V

policy, V = value_iteration(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Value Function:")
print(V)
print("")

print("Reshaped Grid Policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(np.reshape(np.argmax(policy, axis=1), env.desc.shape))
