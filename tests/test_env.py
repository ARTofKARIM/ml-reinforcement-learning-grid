"""Tests for gridworld environment."""
import unittest
from src.environment import GridWorld
from src.q_learning import QLearningAgent

class TestGridWorld(unittest.TestCase):
    def test_reset(self):
        env = GridWorld(grid_size=4, start=(0,0), goal=(3,3))
        state = env.reset()
        self.assertEqual(state, (0, 0))

    def test_step(self):
        env = GridWorld(grid_size=4, start=(0,0), goal=(3,3))
        env.reset()
        state, reward, done = env.step(1)  # DOWN
        self.assertEqual(state, (1, 0))
        self.assertFalse(done)

    def test_goal(self):
        env = GridWorld(grid_size=2, start=(0,0), goal=(1,1))
        env.reset()
        env.step(1)
        _, _, done = env.step(3)
        self.assertTrue(done)

class TestQLearning(unittest.TestCase):
    def test_action_selection(self):
        agent = QLearningAgent(16, 4, epsilon=0)
        agent.q_table[0, 2] = 10
        self.assertEqual(agent.select_action(0), 2)

if __name__ == "__main__":
    unittest.main()
