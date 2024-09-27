import gymnasium as gym
import numpy as np
import random
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, environment, learning_rate=0.1, gamma=0.95, n_components=100, epsilon=0.1):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.eps = epsilon

        feature_sample = np.array([environment.observation_space.sample() for x in range(10000)])
        self.scaler = StandardScaler()
        feature_sample = self.scaler.fit_transform(feature_sample)

        self.RBF = FeatureUnion([
            ("rbf0", RBFSampler(gamma=8.0, n_components=n_components)),
            ("rbf1", RBFSampler(gamma=4.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf5", RBFSampler(gamma=0.25, n_components=n_components))
        ])
        self.RBF.fit(feature_sample)

        self.weights = np.zeros((n_components * 6, self.n_actions))

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate

    def to_feature(self, observation):
        if len(observation.shape) < 2:
            return self.RBF.transform(self.scaler.transform(observation[np.newaxis, :]))
        else:
            return self.RBF.transform(self.scaler.transform(observation))

    def predict_action(self, observation):
        if random.random() > self.eps:
            state = self.to_feature(observation)
            q = np.matmul(state, self.weights)
            return np.argmax(q)
        else:
            return random.randint(0, self.n_actions-1)

    def get_Q(self, state, action):
        return state.dot(self.weights[:, action])

    def fit(self, observation, action, reward, terminated, truncated, new_observation, new_action):
        new_state = self.to_feature(new_observation)
        state = self.to_feature(observation)
        q_current = self.get_Q(state, action)
        q_future = self.get_Q(new_state, new_action)

        # Figure out target, TD error and gradient
        target = reward + (0 if terminated else self.gamma * q_future)
        gradient = (target - q_current).dot(state)

        self.weights[:, action] += self.lr * gradient.squeeze()


# Create environment
env = gym.make("MountainCar-v0")
agent = Agent(env, learning_rate=0.01, gamma=1, n_components=100, epsilon=0.1)

EPISODES, AVG_LEN, TESTS = 1000, 25, 50
history, avg_history = [], []

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]
    steps = 0
    while True:

        action = agent.predict_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_action = agent.predict_action(new_state)
        agent.fit(state, action, int(reward), terminated, truncated, new_state, new_action)

        state = new_state
        steps += 1

        if terminated or truncated:
            break

    history.append(steps)
    avg_score = np.mean(history[-AVG_LEN:])
    avg_history.append(avg_score)

    if (e+1) % AVG_LEN == 0:
        print(f"Episode {e+1}: avg number of steps was {sum(history[-AVG_LEN:])/AVG_LEN:6.2f}")

plt.plot(avg_history)
plt.show()

print("----------------------------------")
print("------------ TESTING -------------")
for e in range(TESTS):
    state = env.reset()[0]
    steps = 0
    while True:
        action = agent.predict_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1

        if terminated or truncated:
            break

    history.append(steps)

    print(f"Test {e+1}: avg number of steps was {sum(history)/len(history):6.2f}")

print("----------------------------------")