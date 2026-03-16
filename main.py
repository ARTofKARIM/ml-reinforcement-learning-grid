"""Main entry for RL gridworld."""
import yaml
from src.environment import GridWorld
from src.q_learning import QLearningAgent
from src.sarsa import SARSAAgent
from src.trainer import RLTrainer
from src.visualization import RLVisualizer

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    ec = config["environment"]
    env = GridWorld(ec["grid_size"], tuple(ec["start"]), tuple(ec["goal"]), ec["obstacles"], ec["rewards"])
    results = {}
    qc = config["agents"]["q_learning"]
    q_agent = QLearningAgent(env.n_states, env.n_actions, qc["alpha"], qc["gamma"], qc["epsilon"], qc["epsilon_decay"])
    trainer = RLTrainer(env, q_agent, "q_learning")
    results["Q-Learning"] = trainer.train(qc["episodes"])
    sc = config["agents"]["sarsa"]
    s_agent = SARSAAgent(env.n_states, env.n_actions, sc["alpha"], sc["gamma"], sc["epsilon"], sc["epsilon_decay"])
    trainer2 = RLTrainer(env, s_agent, "sarsa")
    results["SARSA"] = trainer2.train(sc["episodes"])
    viz = RLVisualizer()
    viz.plot_rewards(results)
    viz.plot_grid(env, q_agent.get_policy(), title="Q-Learning Policy")
    print("Training complete.")

if __name__ == "__main__":
    main()
