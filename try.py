import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from gymnasium import spaces

# Custom Lunar Lander environment with fuel management
class LunarLanderEnvWrapper(gym.Env):
    def __init__(self):
        super(LunarLanderEnvWrapper, self).__init__()
        self.env = gym.make('LunarLander-v2')  # Base environment
        self.state = None

        # Fuel parameters - custom modification
        self.fuel_limit = 200  # Maximum fuel amount
        self.current_fuel = self.fuel_limit
        self.prev_fuel = self.fuel_limit
        self.fuel_consumption_rate = 1  # Amount of fuel consumed per thrust action


        self.action_space = self.env.action_space

    def reset(self):
        self.state, info = self.env.reset()
        self.current_fuel = self.fuel_limit
        # return np.append(self.state, self.current_fuel), info
        return self.state, info
    def step(self, action):
        # Check if there is enough fuel to perform a thrust action
        fuel_penalty = 0
        # if action in [1, 2, 3] and self.current_fuel >= self.fuel_consumption_rate:
        #     self.prev_fuel = self.current_fuel
        #     self.current_fuel -= self.fuel_consumption_rate
        #     fuel_penalty = 0.05
        # elif action in [1, 2, 3] and self.current_fuel < self.fuel_consumption_rate:
        #     # If no fuel, force 'do nothing' action (action 0)
        #     action = 0


        # Perform the action in the environment
        next_state, reward, done, truncated, info = self.env.step(action)
        self.state = next_state  # Update the state

        # check if both legs are on the ground
        # if self.state[6] == 1 and self.state[7] == 1:
        #     done = True

        # if done:
        #     reward += (self.fuel_limit - self.current_fuel) * 0.01  # Adjust penalty rate as needed
        #
        # # # Calculate fuel penalty based on the amount of fuel consumed
        # fuel_penalty = (self.prev_fuel - self.current_fuel) * 0.01  # Adjust penalty rate as needed
        # print(f"prev fule : {self.prev_fuel}, current fuel : {self.current_fuel}, fule_penalty: {fuel_penalty}")
        # reward -= fuel_penalty

        # Check if the episode should terminate due to fuel depletion
        # if self.current_fuel <= 0:
        #     done = True
        #     reward -= 50  # Penalize for running out of fuel
        #     info['termination_reason'] = 'out_of_fuel'  # Add reason to info
        #     print(f"prev reward: {reward+50}, current reward: {reward}")

        # Append fuel level to the state
        # next_state_with_fuel = np.append(next_state, self.current_fuel)
        #
        # reward -= fuel_penalty
        return next_state, reward, done, truncated, info
        # return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()


class LunarLanderEnvFuel(gym.Env):
    def __init__(self):
        super(LunarLanderEnvFuel, self).__init__()
        self.env = gym.make('LunarLander-v2')  # Base environment
        self.state = None

        # Fuel parameters - custom modification
        self.fuel_limit = 100  # Maximum fuel amount
        self.current_fuel = self.fuel_limit
        self.prev_fuel = self.fuel_limit
        self.fuel_consumption_rate = 1  # Amount of fuel consumed per thrust action

        self.action_space = self.env.action_space

    def step(self, action):
        # Check if there is enough fuel to perform a thrust action
        if action in [1, 2, 3] and self.current_fuel >= self.fuel_consumption_rate:
            self.prev_fuel = self.current_fuel
            self.current_fuel -= self.fuel_consumption_rate
        elif action in [1, 2, 3] and self.current_fuel < self.fuel_consumption_rate:
            # If no fuel, force 'do nothing' action (action 0)
            action = 0

        # Perform the action in the environment
        next_state, reward, done, truncated, info = self.env.step(action)
        self.state = next_state  # Update the state

        return next_state, reward, done, truncated, info


    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def reset(self):
        self.state, info = self.env.reset()
        self.current_fuel = self.fuel_limit
        return self.state, info

# Neural Network for DQN
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
    def __init__(self, state_dim, action_dim, action_space, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, lr=0.001, memory_capacity=10000):
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
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
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
def train_dqn(env, agent, num_episodes=4200, update_target_every=10, max_steps_per_episode=200):
    all_rewards = []  # List to store total rewards for each episode

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

        all_rewards.append(total_reward)  # Store total reward for this episode

        if episode % update_target_every == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    # Save the trained model
    agent.save_model('dqn_lunarlander.pth')
    env.close()

    return all_rewards

# Hyperparameter tuning using Random Search
def random_search(env, num_trials=10):
    # Define hyperparameter ranges
    lr_range = [0.0001, 0.001, 0.01]
    gamma_range = [0.95, 0.99, 0.999]
    epsilon_decay_range = [0.995, 0.99, 0.98]
    batch_size_range = [32, 64, 128]
    memory_capacity_range = [5000, 10000, 20000]

    best_avg_reward = -float('inf')
    best_hyperparams = {}

    for trial in range(num_trials):
        # Randomly sample hyperparameters
        lr = random.choice(lr_range)
        gamma = random.choice(gamma_range)
        epsilon_decay = random.choice(epsilon_decay_range)
        batch_size = random.choice(batch_size_range)
        memory_capacity = random.choice(memory_capacity_range)

        print(f"Trial {trial + 1}: LR={lr}, Gamma={gamma}, Epsilon Decay={epsilon_decay}, Batch Size={batch_size}, Memory Capacity={memory_capacity}")

        # Create new agent with sampled hyperparameters
        state_dim = env.env.observation_space.shape[0]
        action_dim = env.action_space.n
        action_space = env.action_space
        agent = DQNAgent(state_dim, action_dim, action_space, batch_size=batch_size, gamma=gamma, epsilon_decay=epsilon_decay, lr=lr, memory_capacity=memory_capacity)

        # Train agent and get average reward
        rewards = train_dqn(env, agent)
        avg_reward = np.mean(rewards[-100:])  # Average reward over the last 100 episodes

        print(f"Average Reward for Trial {trial + 1}: {avg_reward}")

        # Check if this is the best set of hyperparameters
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_hyperparams = {
                'lr': lr,
                'gamma': gamma,
                'epsilon_decay': epsilon_decay,
                'batch_size': batch_size,
                'memory_capacity': memory_capacity
            }

    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Average Reward: {best_avg_reward}")

def test_dqn(env, agent, num_episodes=100, max_steps_per_episode=200):
    avg_reward = 0
    rewards = []
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
        avg_reward += total_reward
        rewards.append(total_reward)

    avg_reward /= num_episodes
    print(f'Average Test Reward: {avg_reward}')
    env.close()
    return rewards


def plot_performance(rewards_classic, rewards_fuel, window=50):
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_rewards_classic = moving_average(rewards_classic, window)
    smoothed_rewards_fuel = moving_average(rewards_fuel, window)

    plt.figure(figsize=(14, 6))

    # Smoothed Line Plot
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_fuel, label='Fuel Env', color='r')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Rewards Over Time')
    plt.legend()

    # Box Plot for Distribution of Rewards
    plt.subplot(1, 2, 2)
    plt.boxplot([rewards_classic, rewards_fuel], labels=['Classic Env', 'Fuel Env'])
    plt.ylabel('Rewards')
    plt.title('Distribution of Rewards')

    plt.tight_layout()
    plt.savefig('performance_comparison.png')

    # Additional Plots: Histogram and Bar Plot
    plt.figure(figsize=(14, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(rewards_classic, bins=30, alpha=0.5, label='Classic Env')
    plt.hist(rewards_fuel, bins=30, alpha=0.5, label='Fuel Env')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Histogram of Rewards')
    plt.legend()

    # Bar Plot of Average Rewards
    avg_rewards = [np.mean(rewards_classic), np.mean(rewards_fuel)]
    plt.subplot(1, 2, 2)
    plt.bar(['Classic Env', 'Fuel Env'], avg_rewards, color=['b', 'r'])
    plt.ylabel('Average Reward')
    plt.title('Average Rewards Comparison')

    plt.tight_layout()
    plt.savefig('additional_plots.png')

    #TODO Gal added this section VVV
def plot_combined_moving_average(rewards_classic, rewards_fuel, window=50):
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data  # Return the original data if too short
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_rewards_classic = moving_average(rewards_classic, window)
    smoothed_rewards_fuel = moving_average(rewards_fuel, window)

    plt.figure(figsize=(10, 6))
    
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_fuel, label='Fuel Env', color='r')
    
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Combined Moving Averages of Rewards')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('combined_moving_average.png')
    plt.show()


# Main function
def main():
    env_classic = gym.make('LunarLander-v2')  # Using base environment
    env_fuel = LunarLanderEnvFuel()  # Using custom environment with fuel

    # random_search(env)  # Hyperparameter tuning

    state_dim = env_classic.observation_space.shape[0]
    action_dim = env_classic.action_space.n
    action_space = env_classic.action_space
    agent = DQNAgent(state_dim, action_dim, action_space)

    # Train the agent
    # print("Training the agent...")
    # train_dqn(env_classic, agent)

    # Load the trained model
    agent.load_model('dqn_lunarlander_classic.pth')

    # Test the agent on the classic environment
    print("Testing the trained agent on the classic environment...")
    rewards_classic = test_dqn(env_classic, agent, num_episodes=1000)

    # Test the agent on the fuel-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    rewards_fuel = test_dqn(env_fuel, agent, num_episodes=1000)

    # Plot the rewards
    plot_performance(rewards_classic, rewards_fuel)
    plot_combined_moving_average(rewards_classic, rewards_fuel)

if __name__ == "__main__":
    main()

