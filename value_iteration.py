import numpy as np


def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Perform value iteration on a Gymnasium environment.

    Args:
        env: The Gymnasium environment (assumed to be FrozenLake).
        gamma: The discount factor.
        theta: A threshold for stopping the value iteration.

    Returns:
        policy: The optimal policy.
        V: The value function for each state.
    """
    # Initialize value function
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        # Update each state's value
        for state in range(env.observation_space.n):
            # Compute the maximum expected value over all possible actions
            v = V[state]
            new_v = max(
                sum(prob * (reward + gamma * V[next_state])
                    for prob, next_state, reward, _ in env.P[state][action])
                for action in range(env.action_space.n)
            )
            # Update the value function
            V[state] = new_v
            # Update the delta
            delta = max(delta, abs(v - new_v))

        # Check for convergence
        if delta < theta:
            break

    # Extract the optimal policy
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        # Select the action with the highest expected value
        policy[state] = np.argmax([
            sum(prob * (reward + gamma * V[next_state])
                for prob, next_state, reward, _ in env.P[state][action])
            for action in range(env.action_space.n)
        ])

        
    return policy, V
