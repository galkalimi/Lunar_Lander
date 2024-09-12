import torch
import torch.nn as nn
import torch.optim as optim
import random
from ReplayBuffer import ReplayBuffer
from DQN import DQN

class DQNAgent:
 
    def __init__(self, state_dim, action_dim, action_space, batch_size=64,
                gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                epsilon_decay=0.995, lr=0.001, memory_capacity=10000,
                dropout_prob=0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.memory = ReplayBuffer(memory_capacity)
        self.policy_net = DQN(state_dim, action_dim, dropout_prob=dropout_prob)
        self.target_net = DQN(state_dim, action_dim, dropout_prob=dropout_prob)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = self.action_space.sample()  # Choose a random action
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                action = self.policy_net(state).argmax().item()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the replay buffer
        transitions = self.memory.sample(self.batch_size)

        # Unpack the batch into separate tensors
        batch = list(zip(*transitions))
        states = torch.stack(batch[0])
        actions = torch.stack(batch[1])
        rewards = torch.stack(batch[2])
        next_states = torch.stack(batch[3])
        dones = torch.stack(batch[4])

        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_net(next_states).max(1)[0]
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
        loss = self.criterion(state_action_values, expected_state_action_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run(self, env, num_iterations, stop_event):
        for episode in range(num_iterations):
            # Reset environment
            state, info = env.reset()
            done = False
            if stop_event and stop_event.is_set():
                break

            while not done or truncated:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                env.render()
                state = next_state

        env.close()
