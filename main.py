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

def test_pid(env, controller, num_episodes=100, max_steps_per_episode=200):
    avg_reward = 0
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            action = controller.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward

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

def pid_plot_performance(rewards_classic, rewards_fuel, rewards_wind, rewards_gravity, window=50):
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
    plt.title('PID Smoothed Rewards Over Time Across Different Environments')
    plt.legend()
    plt.show()

def plot_combined_performance(dqn_rewards_classic, dqn_rewards_fuel, dqn_rewards_wind, dqn_rewards_gravity,
                              pid_rewards_classic, pid_rewards_fuel, pid_rewards_wind, pid_rewards_gravity,
                              window=50, save_files=False, filenames=None, united_filename_dqn=None, united_filename_pid=None):
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Smoothed rewards for DQN
    smoothed_dqn_classic = moving_average(dqn_rewards_classic, window)
    smoothed_dqn_fuel = moving_average(dqn_rewards_fuel, window)
    smoothed_dqn_wind = moving_average(dqn_rewards_wind, window)
    smoothed_dqn_gravity = moving_average(dqn_rewards_gravity, window)

    # Smoothed rewards for PID
    smoothed_pid_classic = moving_average(pid_rewards_classic, window)
    smoothed_pid_fuel = moving_average(pid_rewards_fuel, window)
    smoothed_pid_wind = moving_average(pid_rewards_wind, window)
    smoothed_pid_gravity = moving_average(pid_rewards_gravity, window)

    # Create and save individual plots
    env_titles = ['Classic Environment', 'Fuel Environment', 'Wind Environment', 'Gravity Environment']
    dqn_rewards = [smoothed_dqn_classic, smoothed_dqn_fuel, smoothed_dqn_wind, smoothed_dqn_gravity]
    pid_rewards = [smoothed_pid_classic, smoothed_pid_fuel, smoothed_pid_wind, smoothed_pid_gravity]
    colors = ['b', 'r', 'g', 'm']

    if save_files and filenames:
        for i in range(4):
            plt.figure(figsize=(7, 6))
            plt.plot(dqn_rewards[i], label=f'DQN - {env_titles[i]}', color=colors[i])
            plt.plot(pid_rewards[i], label=f'PID - {env_titles[i]}', color=colors[i], linestyle='--')
            plt.title(env_titles[i])
            plt.xlabel('Episodes')
            plt.ylabel('Smoothed Reward')
            plt.legend()
            plt.savefig(filenames[i])
            plt.close()

    # Create a united graph for all DQN environments
    plt.figure(figsize=(14, 8))
    plt.plot(smoothed_dqn_classic, label='DQN - Classic Env', color='b')
    plt.plot(smoothed_dqn_fuel, label='DQN - Fuel Env', color='r')
    plt.plot(smoothed_dqn_wind, label='DQN - Wind Env', color='g')
    plt.plot(smoothed_dqn_gravity, label='DQN - Gravity Env', color='m')
    plt.title('DQN Performance Across All Environments')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    if save_files and united_filename_dqn:
        plt.savefig(united_filename_dqn)
    plt.show()

    # Create a united graph for all PID environments
    plt.figure(figsize=(14, 8))
    plt.plot(smoothed_pid_classic, label='PID - Classic Env', color='b', linestyle='--')
    plt.plot(smoothed_pid_fuel, label='PID - Fuel Env', color='r', linestyle='--')
    plt.plot(smoothed_pid_wind, label='PID - Wind Env', color='g', linestyle='--')
    plt.plot(smoothed_pid_gravity, label='PID - Gravity Env', color='m', linestyle='--')
    plt.title('PID Performance Across All Environments')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    if save_files and united_filename_pid:
        plt.savefig(united_filename_pid)
    plt.show()

def plot_classic_comparisons(rewards_classic, rewards_fuel, rewards_wind, rewards_gravity, window=50, save_files=False, filenames=None):
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Smoothed rewards
    smoothed_rewards_classic = moving_average(rewards_classic, window)
    smoothed_rewards_fuel = moving_average(rewards_fuel, window)
    smoothed_rewards_wind = moving_average(rewards_wind, window)
    smoothed_rewards_gravity = moving_average(rewards_gravity, window)

    # Plot Classic vs Fuel
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_fuel, label='Fuel Env', color='r')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Classic vs Fuel Environment')
    plt.legend()
    if save_files and filenames:
        plt.savefig(filenames[0])

    # Plot Classic vs Wind
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_wind, label='Wind Env', color='g')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Classic vs Wind Environment')
    plt.legend()
    if save_files and filenames:
        plt.savefig(filenames[1])

    # Plot Classic vs Gravity
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_gravity, label='Gravity Env', color='m')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title('Classic vs Gravity Environment')
    plt.legend()
    if save_files and filenames:
        plt.savefig(filenames[2])

# Main function
def main():
    env_classic = LunarLanderEnvWrapper()
    env_fuel = LunarLanderEnvFuel()  # Using custom environment with fuel
    env_wind = LunarLanderEnvWrapper(gravity=(0,-10), enable_wind=True, wind_power=20.0)  # Using custom environment with wind
    env_gravity = LunarLanderEnvWrapper(gravity=(0,-20))  # Using custom environment with different gravity

    # random_search(env)  # Hyperparameter tuning
    state_dim = env_classic.observation_space.shape[0]
    action_dim = env_classic.action_space.n
    action_space = env_classic.action_space
    agent = DQNAgent(state_dim, action_dim, action_space)

    # Train the agent
    print("Training the agent...")
    # train_dqn(env_classic, agent)

    pid_classic_controller = LunarLanderPIDController(env_classic)
    pid_fuel_controller = LunarLanderPIDController(env_fuel)
    pid_wind_controller = LunarLanderPIDController(env_wind)
    pid_gravity_controller = LunarLanderPIDController(env_gravity)
    # pid_controller.run()

    # Load the trained model
    agent.load_model('dqn_lunarlander_classic.pth')

    # Test the agent on the classic environment
    print("Testing the trained agent on the classic environment...")
    dqn_rewards_classic = test_dqn(env_classic, agent, num_episodes=1000)
    pid_rewards_classic = test_pid(env_classic, pid_classic_controller, num_episodes=1000)

    # Test the agent on the fuel-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    dqn_rewards_fuel = test_dqn(env_fuel, agent, num_episodes=1000)
    pid_rewards_fuel = test_pid(env_classic, pid_fuel_controller, num_episodes=1000)

    # Test the agent on the wind-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    dqn_rewards_wind = test_dqn(env_wind, agent, num_episodes=1000)
    pid_rewards_wind = test_pid(env_classic, pid_wind_controller, num_episodes=1000)

    # Test the agent on the gravity-modified environment
    print("Testing the trained agent on the fuel-modified environment...")
    dqn_rewards_gravity = test_dqn(env_gravity, agent, num_episodes=1000)
    pid_rewards_gravity = test_pid(env_classic, pid_gravity_controller, num_episodes=1000)

    # Plot the rewards  
    plot_combined_performance(dqn_rewards_classic, dqn_rewards_fuel, dqn_rewards_wind, dqn_rewards_gravity,
                          pid_rewards_classic, pid_rewards_fuel, pid_rewards_wind, pid_rewards_gravity,
                          save_files=True, filenames=['classic_comparison.png', 'fuel_comparison.png', 'wind_comparison.png', 'gravity_comparison.png'],
                          united_filename_dqn='all_dqn_comparison.png', united_filename_pid='all_pid_comparison.png')
    plot_classic_comparisons(dqn_rewards_classic, dqn_rewards_fuel, dqn_rewards_wind, dqn_rewards_gravity,
                         save_files=True, filenames=['classic_vs_fuel.png', 'classic_vs_wind.png', 'classic_vs_gravity.png'])
    

if __name__ == "__main__":
    main()