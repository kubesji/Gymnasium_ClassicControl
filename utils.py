import torch
import matplotlib.pyplot as plt


SUCCESS = '\033[1m\033[92msucceded\033[0m'
FAILURE = '\033[1m\033[91m  failed\033[0m'

def to_tensor(action, reward, state_new, done, device, action_type):
    action = torch.tensor(action, dtype=action_type, device=device).unsqueeze(0)
    reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
    state_new = torch.tensor(state_new, dtype=torch.float32, device=device).unsqueeze(0) if not done else None

    return action, reward, state_new


def state_to_tensor(state, device):
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

def plot(history, title=None, xlabel=None, ylabel=None, legend=None):
    if all(isinstance(i, list) for i in history):
        # Multiple subplots
        for i, h in enumerate(history):
            plt.plot(h, linewidth=2, label=legend[i] if legend else None)
    else:
        # Single plot
        plt.plot(history, linewidth=2)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.show()