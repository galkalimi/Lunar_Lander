import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium import spaces

from ReplayBuffer import ReplayBuffer
from LunarLanderEnvWrapper import LunarLanderEnvWrapper

#DQN class
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # Increased number of neurons
        self.fc2 = nn.Linear(256, 256)  # Increased number of neurons
        self.fc3 = nn.Linear(256, 128)  # Additional layer
        self.fc4 = nn.Linear(128, action_dim)  # Additional layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x