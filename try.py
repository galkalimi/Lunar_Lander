import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium import spaces

# Custom Lunar Lander environment with fuel management
class LunarLanderEnvWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make('LunarLander-v2', render_mode="human")  # Base environment
        self.state = None

        # Fuel parameters - custom modification
        self.fuel_limit = 1300  # Maximum fuel amount
        self.current_fuel = self.fuel_limit
        self.fuel_consumption_rate = 5  # Amount of fuel consumed per thrust action

        # Update observation space to include fuel level
        self.observation_space = spaces.Box(
            low=np.append(self.env.observation_space.low, 0),
            high=np.append(self.env.observation_space.high, self.fuel_limit),
            dtype=np.float32
        )

        self.action_space = self.env.action_space

    def reset(self):
        self.state, info = self.env.reset()
        self.current_fuel = self.fuel_limit
        return np.append(self.state, self.current_fuel), info

    def step(self, action):
        # Check if there is enough fuel to perform a thrust action
        if action in [1, 2, 3] and self.current_fuel >= self.fuel_consumption_rate:
            self.current_fuel -= self.fuel_consumption_rate
        elif action in [1, 2, 3] and self.current_fuel < self.fuel_consumption_rate:
            # If no fuel, force 'do nothing' action (action 0)
            action = 0

        # Perform the action in the environment
        next_state, reward, done, truncated, info = self.env.step(action)

        # If fuel runs out, apply penalty but let the episode continue
        if self.current_fuel < self.fuel_consumption_rate and action == 0:
            done = True
            reward -= 50  # Penalize for running out of fuel

        # Append fuel level to the state
        next_state_with_fuel = np.append(next_state, self.current_fuel)

        return next_state_with_fuel, reward, done, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, action_space):
        self.action_dim = action_dim
        self.action_space = action_space
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()  # Choose a random action
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            return self.policy_net(state).argmax().item()

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

        # Compute the predicted Q-values for the current states
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values for the next states using the target network
        next_state_values = self.target_net(next_states).max(1)[0]

        # Compute the expected Q-values for the current states
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Compute the loss between the predicted Q-values and the expected Q-values
        loss = self.criterion(state_action_values, expected_state_action_values.detach())

        # Perform gradient descent to update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        """Save the policy network's weights."""
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """Load the policy network's weights."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Training function
def train_dqn(env, agent, num_episodes=1000, update_target_every=10, max_steps_per_episode=200):
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            agent.memory.push((torch.tensor(state, dtype=torch.float32),
                               torch.tensor(action),
                               torch.tensor(reward, dtype=torch.float32),
                               torch.tensor(next_state, dtype=torch.float32),
                               torch.tensor(done, dtype=torch.float32)))

            agent.optimize_model()
            state = next_state

            if done or truncated:
                break

        if episode % update_target_every == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    # Save the trained model
    agent.save_model('dqn_lunarlander.pth')
    env.close()


def test_dqn(env, agent, num_episodes=10, max_steps_per_episode=200):
    agent.epsilon = 0.0  # Disable exploration for testing
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        print(f'Test Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    env.close()


# Main function
def main():
    env = LunarLanderEnvWrapper()  # Using custom environment with fuel
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_space = env.action_space
    agent = DQNAgent(state_dim, action_dim, action_space)

    # Train the agent
    # print("Training the agent...")
    # train_dqn(env, agent)

    # Load the trained model
    agent.load_model('dqn_lunarlander.pth')

    # Test the agent
    print("Testing the trained agent...")
    test_dqn(env, agent)

if __name__ == "__main__":
    main()

