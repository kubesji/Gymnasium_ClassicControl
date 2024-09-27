import gymnasium as gym
import numpy as np
from dueling_ddqn_torch import Agent
import matplotlib.pyplot as plt
from utils import to_tensor, state_to_tensor, SUCCESS, FAILURE, plot

env_name = "MountainCar-v0" # "Acrobot-v1", "CartPole-v1", "MountainCar-v0"

# Create environment
env = gym.make(env_name)

agent = Agent(0.99, 1e-3, env.action_space.n, env.observation_space.shape, 100000, 64,
              env_name, eps_dec=0.05, update_freq=100)

EPISODES, AVG_LEN = 500, 25
history, avg_history = [], []

print(f"-------- LEARNING --------")
for e in range(EPISODES):
    state = state_to_tensor(env.reset()[0], agent.device)
    steps = 0

    while True:
        action = agent.training_action(state)
        steps += 1

        new_state, reward, terminated, truncated, _ = env.step(action)
        action, reward, new_state = to_tensor(action, reward, new_state, terminated, agent.device)

        agent.store(state, action, reward, terminated, new_state)
        agent.fit()

        state = new_state

        if terminated or truncated:
            break

    history.append(steps)
    avg_score = np.mean(history[-AVG_LEN:])
    avg_history.append(avg_score)

    success = not truncated if env_name != "CartPole-v1" else truncated

    print(f"Episode {e+1:4d} {SUCCESS if success else FAILURE} after {steps:3d}. "
          f"Avg # of steps was {avg_score:6.2f}. Eps={agent.epsilon:.2f}")

    if env_name == "CartPole-v1" and avg_score >= 475:
        break

    agent.decrement_epsilon()

plot(avg_history, title=f'{env_name} - Dueling DDQN', xlabel='Episode', ylabel='# of steps')
