import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from reinforcement_learning.model import QNetwork


class DQNCoreAgent:
    """
    Minimal DQN core logic:
    - build local & target networks
    - epsilon-greedy action selection
    - replay buffer
    - soft update target
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 1e-3,
        update_every: int = 4,
        device: str | None = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed + 1).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size, seed, self.device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            if experiences is None:
                return
            self.learn(experiences)

    def act(self, state, eps: float = 0.1):
        state_t = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            qvals = self.qnetwork_local(state_t)
        self.qnetwork_local.train()

        if random.random() < eps:
            return random.randrange(self.action_size)
        return int(torch.argmax(qvals, dim=1).item())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Q targets for next states
        with torch.no_grad():
            q_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (self.gamma * q_next * (1 - dones))

        # Q expected
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for t_param, l_param in zip(target_model.parameters(), local_model.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)


Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        random.seed(int(seed))

    def add(self, state, action, reward, next_state, done):
        if action is None:
            return
        exp = Experience(
            np.array(state, dtype=np.float32),
            np.array([action], dtype=np.int64),
            np.array([reward], dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array([done], dtype=np.float32)
        )
        self.memory.append(exp)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in batch])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch])).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# Alias agar import DQNAgent tetap bekerja
DQNAgent = DQNCoreAgent