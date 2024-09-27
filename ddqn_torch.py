import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import Memory, Transition

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, device, fc1=256, fc2=256, model_dir='models'):
        super(DuelingDeepQNetwork, self).__init__()
        self.model_file = os.path.join(model_dir, name)

        self.fc1 = nn.Linear(*input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        out = self.out(fc2)

        return out

    def save(self):
        print('... saving model ...')
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        print('... loading model ...')
        self.load_state_dict(torch.load(self.model_file))

class Agent:
    def __init__(self, gamma, lr, n_actions, input_dims,
                 mem_size, batch_size, name, fc1=256, fc2=256, epsilon=1,
                 eps_min=0.01, eps_dec=0.05, update_freq=100, model_dir='models'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.update_every = update_freq
        self.model_dir = model_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = Memory(mem_size)

        self.q_policy = DuelingDeepQNetwork(self.lr, self.n_actions, self.input_dims, name + '_dqn_q_policy',
                                            self.device, fc1=fc1, fc2=fc2)

        self.q_target = DuelingDeepQNetwork(self.lr, self.n_actions, self.input_dims, name + '_dqn_q_target',
                                            self.device, fc1=fc1, fc2=fc2)
        self.replace_target_network()

    def training_action(self, state):
        if np.random.random() > self.epsilon:
            action = self.select_action(state)
        else:
            action = np.random.choice(self.action_space)

        return action

    def select_action(self, state):
        value = self.q_policy.forward(state)
        action = torch.argmax(value).item()
        return action

    def store(self, state, action, reward, state_new, done):
        self.memory.store(state, action, reward, state_new, done)

    def replace_target_network(self):
        self.q_target.load_state_dict(self.q_policy.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        self.q_policy.save()
        self.q_target.save()

    def load_models(self):
        self.q_policy.load()
        self.q_target.load()

    def optimistic_bias(self, env, batch=64, episodes=100):
        print("Biasing agent to increase exploration")
        for _ in range(episodes):
            samples = torch.tensor([env.observation_space.sample() for i in range(batch)]).to(self.device)
            actions = torch.tensor([random.randint(0, self.n_actions-1) for i in range(batch)]).to(self.device)
            target = torch.tensor([10.0] * batch).to(self.device)

            self.q_policy.optimizer.zero_grad()

            indices = np.arange(self.batch_size)
            q_values = self.q_policy(samples)[indices, actions]

            loss = self.q_policy.loss(q_values, target).to(self.device)
            loss.backward()
            self.q_policy.optimizer.step()

        self.replace_target_network()
        print("Biasing done")


    def fit(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_policy.optimizer.zero_grad()

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.new_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.new_state if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        indices = np.arange(self.batch_size)

        q_values = self.q_policy(states)[indices, actions]
        q_values_next = torch.zeros((self.batch_size, self.n_actions), device=self.device)
        q_eval_next = torch.zeros((self.batch_size, self.n_actions), device=self.device)
        with torch.no_grad():
            q_values_next[non_final_mask] = self.q_target(non_final_next_states)

        q_eval_next[non_final_mask] = self.q_policy(non_final_next_states)

        max_actions = torch.argmax(q_eval_next, dim=1)

        q_values_next[~non_final_mask] = 0.0
        q_target = rewards + self.gamma*q_values_next[indices, max_actions]

        loss = self.q_policy.loss(q_values, q_target).to(self.device)
        loss.backward()
        self.q_policy.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_every == 0:
            self.replace_target_network()
