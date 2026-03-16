# Reinforcement Learning Gridworld

RL agents (Q-Learning, SARSA) learning to navigate a configurable gridworld environment with obstacles and rewards.

## Architecture
```
ml-reinforcement-learning-grid/
├── src/
│   ├── environment.py  # Configurable gridworld
│   ├── q_learning.py   # Q-Learning agent
│   ├── sarsa.py         # SARSA agent
│   ├── trainer.py       # Training loop
│   └── visualization.py # Reward curves, policy grids
├── config/config.yaml
├── tests/test_env.py
└── main.py
```
## Installation
```bash
git clone https://github.com/mouachiqab/ml-reinforcement-learning-grid.git
cd ml-reinforcement-learning-grid && pip install -r requirements.txt
python main.py
```
## Technologies
- Python 3.9+, NumPy, Gymnasium, matplotlib












