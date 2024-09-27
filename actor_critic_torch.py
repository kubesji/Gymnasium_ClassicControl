import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import Memory, Transition


class Model(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, device, fc1=256, fc2=256, model_dir='models'):
        super(Model, self).__init__()
        self.model_file = os.path.join(model_dir, name)

        self.fc1 = nn.Linear(*input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
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
    def __init__(self, gamma, lr_actor, lr_critic, n_actions, input_dims,
                 mem_size, batch_size, name, fc1=256, fc2=256,
                 eps_min=0.01, eps_dec=0.05,model_dir='models'):
        self.gamma = gamma
        self.epsilon = 1
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.model_dir = model_dir
        self.action_space = [i for i in range(self.n_actions)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = Memory(mem_size)
        self.sample = None

        self.actor = Model(lr_actor, self.n_actions, self.input_dims, name + '_actor',
                           self.device, fc1=fc1, fc2=fc2)
        self.critic = Model(lr_critic, 1, self.input_dims, name + '_critic',
                            self.device, fc1=fc1, fc2=fc2)

    def training_action(self, state):
        probs = F.softmax(self.actor(state), dim=1).squeeze()
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        return action.item(), log_prob

    def store(self, state, action, reward, state_new, done):
        self.memory.store(state, action, reward, state_new, done)
        self.sample = (state, action, reward, state_new, done)

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        self.actor.save()
        self.critic.save()

    def load_models(self):
        self.actor.load()
        self.critic.load()

    def fit(self):

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        (state, log_probs, reward, state_new, done) = self.sample

        #log_probs = torch.tensor(log_probs)
        critic_values_new = self.critic(state_new).squeeze() if not done else torch.tensor(0)
        critic_values = self.critic(state).squeeze()

        delta = reward + self.gamma * critic_values_new - critic_values
        actor_loss = -log_probs * delta
        critic_loss = torch.square(delta)
        loss = torch.mean(actor_loss + critic_loss)

        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
