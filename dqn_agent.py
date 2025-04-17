# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 3)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
        expected_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
