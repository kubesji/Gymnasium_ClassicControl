import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from memory import Memory, Transition


class Actor(nn.Module):
    def __init__(self, lr, n_outputs, input_dims, name, device, fc1=256, fc2=256, model_dir='models'):
        super(Actor, self).__init__()
        self.model_file = os.path.join(model_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, n_outputs)

        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, lr, state_dim, action_dim, name, device, fc1=256, fc2=256, model_dir='models'):
        super(Critic, self).__init__()
        self.model_file = os.path.join(model_dir, name)

        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1 + action_dim, fc2)
        self.fc3 = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, gamma, actor_lr, critic_lr, state_dim, n_outputs, buffer_size, batch_size, tau,
                 noise_decay=0.05, noise_min=0.01, fc_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(actor_lr, n_outputs[0], state_dim[0], "actor", self.device, fc1=fc_size, fc2=fc_size)
        self.actor_target = Actor(actor_lr, n_outputs[0], state_dim[0], "actor_target", self.device,
                                  fc1=fc_size, fc2=fc_size)
        self.critic = Critic(critic_lr, state_dim[0], n_outputs[0], "critic", self.device, fc1=fc_size, fc2=fc_size)
        self.critic_target = Critic(critic_lr, state_dim[0], n_outputs[0], "critic", self.device,
                                    fc1=fc_size, fc2=fc_size)

        self.memory = Memory(length=buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.n_outputs = n_outputs[0]
        self.epsilon = 0.25  # Epsilon = noise here
        self.noise_decay = noise_decay
        self.noise_min = noise_min

        self._update_target_networks(tau=1)  # initialize target networks

    def training_action(self, state):
        action = self.actor(state).detach().cpu().numpy()[0]
        noise = random.uniform(-self.epsilon, self.epsilon)
        return np.clip(action + noise, -1, 1)

    def store(self, state, action, reward, state_new, done):
        self.memory.store(state, action, reward, state_new, done)

    def decrement_epsilon(self):
        self.epsilon = max(self.noise_min, self.epsilon - self.noise_decay)

    def fit(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.new_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.new_state if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        # Update Critic
        self.critic.optimizer.zero_grad()

        target_q_values = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.float32)

        with torch.no_grad():
            next_actions = self.actor_target(non_final_next_states)
            target_q_values[non_final_mask] = self.critic_target(non_final_next_states, next_actions)
            target_q_values = rewards.unsqueeze(1) + target_q_values * self.gamma

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_params(self):
        return (self.actor_target.parameters(), self.actor.parameters(), self.critic_target.parameters(),
                self.critic.parameters())

