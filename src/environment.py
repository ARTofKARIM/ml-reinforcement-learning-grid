"""Gridworld environment for RL agents."""
import numpy as np

class GridWorld:
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, grid_size=8, start=(0,0), goal=(7,7), obstacles=None, rewards=None):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(tuple(o) for o in (obstacles or []))
        self.rewards = rewards or {"step": -0.1, "goal": 10.0, "obstacle": -5.0}
        self.state = self.start
        self.grid = np.zeros((grid_size, grid_size))
        for obs in self.obstacles:
            self.grid[obs] = -1

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        new_r = max(0, min(self.grid_size - 1, self.state[0] + dr))
        new_c = max(0, min(self.grid_size - 1, self.state[1] + dc))
        new_state = (new_r, new_c)

        if new_state in self.obstacles:
            reward = self.rewards["obstacle"]
            new_state = self.state
        elif new_state == self.goal:
            reward = self.rewards["goal"]
        else:
            reward = self.rewards["step"]

        self.state = new_state
        done = self.state == self.goal
        return self.state, reward, done

    def get_state_index(self, state):
        return state[0] * self.grid_size + state[1]

    @property
    def n_states(self):
        return self.grid_size * self.grid_size

    @property
    def n_actions(self):
        return 4
