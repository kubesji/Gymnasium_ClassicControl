import gymnasium as gym
import numpy as np
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, environment, learning_rate=0.1, gamma=0.95, epsilon=0.1, bootstrapping=4,
                 n_steps_digitize=16):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space.shape
        self.shift = -environment.observation_space.low
        self.steps = (environment.observation_space.high - environment.observation_space.low) / n_steps_digitize

        self.Q = defaultdict(lambda: [0 for _ in range(self.n_actions)])

        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon

        self.memory = deque(maxlen=bootstrapping)
        self.bootstrapping = bootstrapping

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

    def fit(self, trailing=False):
        G = 0
        for i in range(len(self.memory)):
            reward = self.memory[i][2]
            G += self.gamma**i * reward

        state, action = self.memory[0][0], self.memory[0][1]
        new_state, new_action = self.memory[-1][5], self.memory[-1][6]
        if self.bootstrapping == len(self.memory):
            G += self.gamma**self.bootstrapping * self.Q[new_state][new_action]

        self.Q[state][action] = (1 - self.lr) * self.Q[state][action] + self.lr * G

    def store(self, state, action, reward, terminated, truncated, new_state, new_action):
        self.memory.append((state, action, reward, terminated, truncated, new_state, new_action))


# Create environment
env = gym.make("Acrobot-v1")

bootstrappings = [1, 2, 4, 8, 16, 32]
results = {}
avg_histories = []
TESTS, EPISODES, AVG_LEN = 250, 10000, 500

for i, b in enumerate(bootstrappings):
    agent = Agent(env, learning_rate=0.1, gamma=0.99, epsilon=0.1, bootstrapping=b, n_steps_digitize=8)
    history, avg_history = [], []

    print(f"-------- LEARNING - B{b} --------")
    for e in range(EPISODES):
        state = agent.box_to_discrete(env.reset()[0])
        steps = 0
        terminated, truncated = False, False
        action = agent.get_training_action(state)

        while not terminated and not truncated:
            steps += 1

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = agent.box_to_discrete(new_state)
            new_action = agent.get_training_action(new_state)

            agent.store(state, action, int(reward), terminated, truncated, new_state, new_action)
            if steps >= agent.bootstrapping:
                agent.fit()

            state = new_state
            action = new_action

        history.append(steps)
        avg_score = np.mean(history[-AVG_LEN:])
        avg_history.append(avg_score)

        if (e+1) % AVG_LEN == 0:
            print(f"Episode {e+1}: avg # of steps was {avg_score}")

    print("----------------------------------\n")

    avg_histories.append(avg_history)

    print(f"----------- TEST - B{b} -----------")
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

    avg = sum(history)/len(history)
    print(f"Average # of steps was {avg}. Unsuccessful efforts: {not_reached}")
    print("----------------------------------\n3")

    results[b] = (avg, not_reached)

for b, r in results.items():
    print(f"Bootstrapping {b}: avg # of steps was {r[0]}, missed goal {r[1]/TESTS*100:.2f} %")

for h, b in zip(avg_histories, bootstrappings):
    plt.plot(h, label = f"b{b}")
plt.legend()
plt.show()