# Description: Replay buffer for DQN agent to store transitions and sample mini-batches for training.
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium import spaces



class ReplayBuffer:
   def __init__(self, capacity):
       self.buffer = deque(maxlen=capacity)

   def push(self, transition):
       self.buffer.append(transition)

   def sample(self, batch_size):
       return random.sample(self.buffer, batch_size)

   def __len__(self):
       return len(self.buffer)
