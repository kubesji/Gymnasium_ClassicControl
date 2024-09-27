import gymnasium as gym
import numpy as np
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, environment, learning_rate=0.1, gamma=0.95, epsilon=0.1, n_steps=16):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space.shape
        self.shift = -environment.observation_space.low
        self.steps = (environment.observation_space.high - environment.observation_space.low) / n_steps

        # Deliberately placing 1 here for optimistic search - unvisited state-action pairs look better
        self.Q = defaultdict(lambda: [1 for _ in range(self.n_actions)])

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon

    def box_to_discrete(self, state):
        return tuple(((state + self.shift) / self.steps).astype(int))

    def predict_action(self, state):
        return np.argmax(self.Q[state])

    def get_training_action(self, state):
        # Take random action
        if random.random() <= self.eps:
            return random.randint(0, self.n_actions-1)
        # Predict best action
        else:
            return self.predict_action(state)

    def fit(self, state, action, reward, terminated, truncated, new_state):
        max_future_q = np.max(self.Q[new_state]) if not terminated else 0
        q_current = self.Q[state][action]

        new_q = (1 - self.lr) * q_current + self.lr * (reward + self.gamma * max_future_q)
        self.Q[state][action] = new_q


# Create environment
env = gym.make("MountainCar-v0")
agent = Agent(env, learning_rate=0.01, gamma=0.99, epsilon=0.1, n_steps=16)

TESTS, EPISODES, AVG_LEN = 50, 200000, 1000
history, avg_history = [], []

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = agent.box_to_discrete(env.reset()[0])
    steps = 0
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = agent.get_training_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        new_state = agent.box_to_discrete(new_state)
        agent.fit(state, action, int(reward), terminated, truncated, new_state)

        state = new_state
        steps += 1

    history.append(steps)
    avg_score = np.mean(history[-AVG_LEN:])
    avg_history.append(avg_score)

    if (e+1) % AVG_LEN == 0:
        print(f"Episode {e+1}: avg number of steps was {sum(history[-AVG_LEN:])/AVG_LEN}")

plt.plot(avg_history)
plt.show()

print("----------------------------------\n\n-------------- TEST --------------")
history.clear()
not_reached = 0
for t in range(TESTS):
    state = agent.box_to_discrete(env.reset()[0])
    steps, terminated, truncated = 0, False, False
    while not terminated and not truncated:
        action = agent.predict_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)

        state = agent.box_to_discrete(new_state)
        steps += 1

    history.append(steps)

    if truncated:
        not_reached += 1

    print(f"Test {t}: agent {'didnt reach the goal.' if truncated else f'reached the goal after {steps}.'}")

print(f"Average # of steps was {sum(history)/len(history)}. Unsuccessful efforts: {not_reached}")

print("----------------------------------")
