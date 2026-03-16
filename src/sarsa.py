"""SARSA agent."""
import numpy as np

class SARSAAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = n_actions

    def select_action(self, state_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action, reward, next_state_idx, next_action, done):
        td_target = reward + (1 - done) * self.gamma * self.q_table[next_state_idx, next_action]
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
