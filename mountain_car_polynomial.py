import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Agent:
    def __init__(self, environment, learning_rate=0.1, gamma=0.95, polynomial_degree=1, epsilon=0.1):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space.shape
        self.eps = epsilon

        self.pol_deg = polynomial_degree

        if polynomial_degree == 0:
            self.weights = np.zeros((self.observation_space[0], self.n_actions))
        else:
            self.weights = np.zeros(((self.pol_deg+1)**self.observation_space[0], self.n_actions))

        feature_sample = np.array([environment.observation_space.sample() for x in range(10000)])
        self.scaler = StandardScaler()
        self.scaler.fit(feature_sample)

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate

    def to_feature(self, observation):
        observation = self.scaler.transform(observation.reshape(1, -1))[0]
        if self.pol_deg == 0:
            return observation
        s = [observation[0]**i * observation[1]**j for i in range(self.pol_deg+1) for j in range(self.pol_deg+1)]
        return np.array(s)

    def predict_action(self, observation):
        if random.random() > self.eps:
            state = self.to_feature(observation)
            q = np.matmul(state, self.weights)
            return np.argmax(q)
        else:
            return random.randint(0, self.n_actions - 1)

    def get_Q(self, state, action):
        return state.dot(self.weights[:, action])

    def fit(self, observation, action, reward, terminated, truncated, new_observation, new_action):
        new_state = self.to_feature(new_observation)
        state = self.to_feature(observation)
        q_current = self.get_Q(state, action)
        q_future = self.get_Q(new_state, new_action)

        # Figure out target, TD error and gradient
        target = reward + (0 if terminated else self.gamma * q_future)
        gradient = (target - q_current) * state

        self.weights[:, action] += self.lr * gradient.squeeze()


# Create environment
env = gym.make("MountainCar-v0", max_episode_steps=250)
agent = Agent(env, learning_rate=0.01, gamma=1, polynomial_degree=10, epsilon=0.1)

EPISODES, AVG_LEN = 2000, 25
history, avg_history = [], []

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]
    # action = agent.predict_action(state)
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