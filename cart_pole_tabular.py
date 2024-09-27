import gymnasium as gym
import numpy as np
import random
from collections import deque, defaultdict


class Agent:
    def __init__(self, environment, learning_rate=0.1, gamma=0.95, epsilon=0.1,
                 n_steps=6):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space.shape
        limits = [2.4, 0.8, 0.418, 3.5]     # Little trick here - although observations 1 and 3 can go to infinity,
                                            # in practise, these don't exceed roughly 0.62 and 3.1
        self.milestones = [list(np.linspace(-l, l, num=n_steps)) for l in limits]
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon

    def observation_to_discrete(self, state):
        discrete = []

        for s, milestones in zip(state, self.milestones):
            for i, m in enumerate(milestones):
                if s <= m:
                    discrete.append(i)
                    break
        #print(discrete)
        return tuple(discrete)

    def predict_action(self, state):
        return np.argmax(self.Q[self.observation_to_discrete(state)])

    def get_training_action(self, state):
        # Take random action
        if random.random() <= self.eps:
            return random.randint(0, self.n_actions-1)
        # Predict best action
        else:
            return self.predict_action(state)

    def fit(self, state, action, reward, terminated, truncated, new_state):
        state_discrete = self.observation_to_discrete(state)
        new_state_discrete = self.observation_to_discrete(new_state)
        max_future_q = np.max(self.Q[new_state_discrete]) if not terminated else 0
        q_current = self.Q[state_discrete][action]

        new_q = (1 - self.lr) * q_current + self.lr * (reward + self.gamma * max_future_q)
        self.Q[state_discrete][action] = new_q



# Create environment
env = gym.make("CartPole-v1", )
agent = Agent(env, learning_rate=0.05, epsilon=0.1, n_steps=24, gamma=0.99)

# Initialise stuff needed for exploration and exploitation
# High number of episodes is needed to fully explore all possible combinations
EPISODES, TESTS = 150000, 50
history = deque(maxlen=1000)

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]

    terminated, truncated = False, False
    steps = 0
    while not terminated and not truncated:
        action = agent.get_training_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        #if terminated:
        #    reward = -1
        agent.fit(state, action, int(reward), terminated, truncated, new_state)

        state = new_state
        steps += 1

    history.append(steps)

    if (e+1) % 1000 == 0:
        print(f"Episode {e+1}: {np.mean(history)} steps")


print("-------------- TEST --------------")
history.clear()
found = 0
for t in range(TESTS):
    state = env.reset()[0]
    steps, terminated, truncated = 0, False, False
    while not terminated and not truncated:
        action = agent.predict_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)

        state = new_state
        steps += 1

    print(f"Test {t + 1}: {steps} steps")
