from re import T
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium import spaces
from DQNAgent import DQNAgent
from LunarLanderEnvFuel import LunarLanderEnvFuel
from ReplayBuffer import ReplayBuffer
from LunarLanderEnvWrapper import LunarLanderEnvWrapper
from DQN import DQN
from LunarLanderPIDController import LunarLanderPIDController

#main

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
def train_dqn_on_multiple_envs(envs, agent, num_episodes=10000, update_target_every=10, max_steps_per_episode=200):
    all_rewards = []  # List to store total rewards across all environments

    for episode in range(num_episodes):
        env = random.choice(envs)  # Pick a random environment for each episode
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
    agent.save_model('dqn_multi_gravities_10_105_115_12.pth')

    # Close all environments
    for env in envs:
        env.close()

    return all_rewards

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

def plot_performance(rewards_classic, rewards_fuel, rewards_wind, rewards_gravity, window=50):
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_rewards_classic = moving_average(rewards_classic, window)
    smoothed_rewards_fuel = moving_average(rewards_fuel, window)
    smoothed_rewards_wind = moving_average(rewards_wind, window)
    smoothed_rewards_gravity = moving_average(rewards_gravity, window)

    plt.figure(figsize=(14, 6))

    # Smoothed Line Plot
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_fuel, label='Fuel Env', color='r')
    plt.plot(smoothed_rewards_wind, label='Wind Env', color='g')
    plt.plot(smoothed_rewards_gravity, label='Gravity Env', color='m')
    
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Rewards Over Time Across Different Environments')
    plt.legend()
    plt.show()

    # # Additional Plots: Histogram and Bar Plot
    # plt.figure(figsize=(14, 6))

    # # Histogram
    # plt.subplot(1, 2, 1)
    # plt.hist(rewards_classic, bins=30, alpha=0.5, label='Classic Env')
    # plt.hist(rewards_fuel, bins=30, alpha=0.5, label='Fuel Env')
    # plt.xlabel('Reward')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Rewards')
    # plt.legend()

    # # Bar Plot of Average Rewards
    # avg_rewards = [np.mean(rewards_classic), np.mean(rewards_fuel)]
    # plt.subplot(1, 2, 2)
    # plt.bar(['Classic Env', 'Fuel Env'], avg_rewards, color=['b', 'r'])
    # plt.ylabel('Average Reward')
    # plt.title('Average Rewards Comparison')

    # plt.tight_layout()
    # plt.savefig('additional_plots.png')

# Main function
def main():
    env_classic = LunarLanderEnvWrapper()
    env_fuel = LunarLanderEnvFuel()  # Using custom environment with fuel
    env_wind = LunarLanderEnvWrapper(gravity=(0,-10), enable_wind=True, wind_power=20.0)  # Using custom environment with wind
    env_gravity = LunarLanderEnvWrapper(gravity=(0,-11))  # Using custom environment with different gravity

    # random_search(env)  # Hyperparameter tuning
    state_dim = env_classic.observation_space.shape[0]
    action_dim = env_classic.action_space.n
    action_space = env_classic.action_space
    agent = DQNAgent(state_dim, action_dim, action_space)

    # Train the agent
    print("Training the agent...")
    # train_dqn(env_classic, agent)

    # env_gravity100 = LunarLanderEnvWrapper(gravity=(0,-10))
    # env_gravity105 = LunarLanderEnvWrapper(gravity=(0,-10.5))
    # env_gravity115 = LunarLanderEnvWrapper(gravity=(0,-11.5))
    # env_gravity120 = LunarLanderEnvWrapper(gravity=(0,-12))
    # envs = [env_gravity100, env_gravity105, env_gravity115, env_gravity120]
    # train_dqn_on_multiple_envs(envs, agent)

    env = gym.make("LunarLander-v2", render_mode="human")
    pid_controller = LunarLanderPIDController(env)
    pid_controller.run()

    # Load the trained model
    # agent.load_model('dqn_lunarlander_classic.pth')
    # agent.load_model('dqn_multi_env.pth')
    # agent.load_model('dqn_lunarlander_classic.pth')

    # Test the agent on the classic environment
    print("Testing the trained agent on the classic environment...")
    rewards_classic = test_dqn(env_classic, agent, num_episodes=1000)

    # Test the agent on the fuel-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    rewards_fuel = test_dqn(env_fuel, agent, num_episodes=1000)

    # Test the agent on the wind-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    rewards_wind = test_dqn(env_wind, agent, num_episodes=1000)

    # Test the agent on the gravity-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    rewards_gravity = test_dqn(env_gravity, agent, num_episodes=1000)

    # Plot the rewards
    plot_performance(rewards_classic, rewards_fuel, rewards_wind, rewards_gravity)
    

if __name__ == "__main__":
    main()