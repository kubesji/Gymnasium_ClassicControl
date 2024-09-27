import gymnasium as gym
import numpy as np
np.float_ = np.float64
import matplotlib.pyplot as plt
from utils import to_tensor, state_to_tensor, SUCCESS, FAILURE, plot
import torch

env_name = "CartPole-v1" # "Acrobot-v1", "CartPole-v1", "MountainCar-v0", Pendulum-v1, 'MountainCarContinuous-v0'
agent_type = "Actor-Critic Buffer" # DDQN, Dealing DDQN, Actor-Critic, DDPG
agent = None
action_type = torch.long

# Create environment
env = gym.make(env_name)

if agent_type == "DDQN":
    from ddqn_torch import Agent
    if env_name == "MountainCar-v0":
        agent = Agent(0.99, 1e-4, env.action_space.n, env.observation_space.shape, 100000, 64,
                      env_name, epsilon=-1, update_freq=200)
        agent.optimistic_bias(env, episodes=10000)
    else:
        agent = Agent(0.99, 1e-4, env.action_space.n, env.observation_space.shape, 100000, 64,
                      env_name, eps_dec=0.05, update_freq=200)
elif agent_type == "Dueling DDQN":
    from dueling_ddqn_torch import Agent
    agent = Agent(0.99, 1e-3, env.action_space.n, env.observation_space.shape, 100000, 64,
                  env_name, eps_dec=0.05, update_freq=100)
elif agent_type == "Actor-Critic":
    from actor_critic_torch import Agent
    agent = Agent(0.99, 1e-4, 5e-4, env.action_space.n, env.observation_space.shape,
                  100000, 64, env_name, eps_dec=0.05, eps_min=0.01, fc1=128, fc2=128)
elif agent_type == "Actor-Critic Continuous":
    from actor_critic_torch_continuous import Agent
    agent = Agent(0.99, 1e-3, 5e-3, env.observation_space.shape,
                  100000, 64, env_name, eps_dec=0.05, eps_min=0.01, fc1=128, fc2=128)
elif agent_type == "DDPG":
    from ddpg_continuous import Agent
    agent = Agent(0.95, 5e-4, 1e-3, env.observation_space.shape, env.action_space.shape,
                  10000, 64, 5e-3, noise_min=0.01, noise_decay=0.01, fc_size=256)
    action_type = torch.float32
else:
    print("No valid agent selected")
    exit(-1)


EPISODES, AVG_LEN = 500, 25
history, avg_history = [], []
prob = None

print(f"-------- LEARNING --------")
for e in range(EPISODES):
    state = state_to_tensor(env.reset()[0], agent.device)
    steps, score = 0, 0

    while True:
        if "Actor-Critic" in agent_type:
            action, prob = agent.training_action(state)
        else:
            action = agent.training_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        action, reward, new_state = to_tensor(action, reward, new_state, terminated, agent.device, action_type)

        agent.store(state, action if "Actor-Critic" not in agent_type else prob, reward, new_state, terminated)

        agent.fit()

        steps += 1
        state = new_state

        if terminated or truncated:
            break


    history.append(score)
    avg_score = np.mean(history[-AVG_LEN:])
    avg_history.append(avg_score)

    success = not truncated if env_name != "CartPole-v1" else truncated
    print(f"Episode {e+1:4d} {SUCCESS if success else FAILURE} after {steps:3d} steps with score {score:7.2f}. "
          f"Avg # score was {avg_score:6.2f}. Eps={agent.epsilon:.2f}")

    if env_name == "CartPole-v1" and avg_score >= 475:
        break

    agent.decrement_epsilon()

plot(avg_history, title=f'{env_name} - {agent_type}', xlabel='Episode', ylabel='Score')
