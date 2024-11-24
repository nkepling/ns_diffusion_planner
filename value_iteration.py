import numpy as np
from itertools import product


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
    QV = np.zeros((env.observation_space.n, env.action_space.n))

    while True:
        delta = 0
        # Update each state's value
        for state, action in product(range(env.observation_space.n), range(env.action_space.n)):
            # Compute the maximum expected value over all possible actions
            q = QV[state, action]

            new_q = sum(prob * (reward + gamma * max(QV[next_state, :]))
                        for prob, next_state, reward, _ in env.P[state][action])

            # Update the qvalue function
            QV[state, action] = new_q

            # Update the delta
            delta = max(delta, abs(q - new_q))

        # Check for convergence
        if delta < theta:
            break

    # Extract the optimal policy
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        # Select the action with the highest expected value
        policy[state] = np.argmax(QV[state, :])

    return policy, QV
