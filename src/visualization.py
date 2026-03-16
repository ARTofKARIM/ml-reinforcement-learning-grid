"""Visualization for RL training."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class RLVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_rewards(self, rewards_dict, window=100, save=True):
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, rewards in rewards_dict.items():
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (smoothed)")
        ax.set_title("Training Rewards Comparison")
        ax.legend()
        if save:
            fig.savefig(f"{self.output_dir}rewards.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_grid(self, env, policy=None, q_table=None, title="", save=True):
        fig, ax = plt.subplots(figsize=(8, 8))
        grid = np.zeros((env.grid_size, env.grid_size))
        for obs in env.obstacles:
            grid[obs] = -1
        grid[env.goal] = 1
        ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
        arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        if policy is not None:
            for r in range(env.grid_size):
                for c in range(env.grid_size):
                    if (r, c) not in env.obstacles and (r, c) != env.goal:
                        idx = env.get_state_index((r, c))
                        ax.text(c, r, arrows[policy[idx]], ha="center", va="center", fontsize=14)
        ax.set_title(title)
        if save:
            fig.savefig(f"{self.output_dir}grid_{title.lower().replace(' ','_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
