from collections import namedtuple, deque
import random


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'new_state', 'terminal'))


class Memory:
    def __init__(self, length):
        self.mem = deque(maxlen=length)

    def store(self, *args):
        self.mem.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)

    def clear(self):
        self.mem.clear()

    def get_all(self):
        return list(self.mem)