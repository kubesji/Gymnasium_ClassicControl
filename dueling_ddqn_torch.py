import os
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
        self.V = nn.Linear(fc2, 1)
        self.A = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        V = self.V(fc2)
        A = self.A(fc2)

        return V, A

    def save(self):
        print('... saving model ...')
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        print('... loading model ...')
        self.load_state_dict(torch.load(self.model_file))


class Agent:
    def __init__(self, gamma, lr, n_actions, input_dims,
                 mem_size, batch_size, name, fc1=256, fc2=256,
                 eps_min=0.01, eps_dec=0.05, update_freq=100, model_dir='models'):
        self.gamma = gamma
        self.epsilon = 1
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

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions, self.input_dims, name + '_dueling_ddqn_q_eval',
                                          self.device, fc1=fc1, fc2=fc2)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions, self.input_dims, name + '_dueling_ddqn_q_next',
                                          self.device, fc1=fc1, fc2=fc2)
        self.replace_target_network()

    def training_action(self, state):
        if np.random.random() > self.epsilon:
            action = self.select_action(state)
        else:
            action = np.random.choice(self.action_space)

        return action

    def select_action(self, state):
        _, advantage = self.q_eval.forward(state)
        action = torch.argmax(advantage).item()
        return action

    def store(self, state, action, reward, state_new, done):
        self.memory.store(state, action, reward, state_new, done)

    def replace_target_network(self):
        self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        self.q_eval.save()
        self.q_next.save()

    def load_models(self):
        self.q_eval.load()
        self.q_next.load()

    def fit(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.new_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.new_state if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval(states)
        V_s_new = torch.zeros((self.batch_size, 1), device=self.device)
        A_s_new = torch.zeros((self.batch_size, self.n_actions), device=self.device)
        with torch.no_grad():
            V_s_new[non_final_mask], A_s_new[non_final_mask] = self.q_next(non_final_next_states)

        V_s_eval = torch.zeros((self.batch_size, 1), device=self.device)
        A_s_eval = torch.zeros((self.batch_size, self.n_actions), device=self.device)
        V_s_eval[non_final_mask], A_s_eval[non_final_mask] = self.q_eval(non_final_next_states)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_new, (A_s_new - A_s_new.mean(dim=1, keepdim=True)))

        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)

        q_next[~non_final_mask] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_every == 0:
            self.replace_target_network()
