"""Training loop for RL agents."""
import numpy as np
from tqdm import trange

class RLTrainer:
    def __init__(self, env, agent, agent_type="q_learning"):
        self.env = env
        self.agent = agent
        self.agent_type = agent_type
        self.episode_rewards = []
        self.episode_lengths = []

    def train(self, n_episodes=5000, max_steps=200):
        for ep in trange(n_episodes, desc=f"Training {self.agent_type}"):
            state = self.env.reset()
            state_idx = self.env.get_state_index(state)
            total_reward = 0
            if self.agent_type == "sarsa":
                action = self.agent.select_action(state_idx)
            for step in range(max_steps):
                if self.agent_type == "q_learning":
                    action = self.agent.select_action(state_idx)
                next_state, reward, done = self.env.step(action)
                next_state_idx = self.env.get_state_index(next_state)
                total_reward += reward
                if self.agent_type == "q_learning":
                    self.agent.update(state_idx, action, reward, next_state_idx, done)
                elif self.agent_type == "sarsa":
                    next_action = self.agent.select_action(next_state_idx)
                    self.agent.update(state_idx, action, reward, next_state_idx, next_action, done)
                    action = next_action
                state_idx = next_state_idx
                if done:
                    break
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step + 1)
            self.agent.decay_epsilon()
        return self.episode_rewards
