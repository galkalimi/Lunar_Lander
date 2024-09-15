from matplotlib import pyplot as plt
import numpy as np

"""
This module provides various functions to plot the performance of reinforcement learning agents across different environments 
and compare them with different controllers such as DQN and PID.

Functions:
    plot_performance(rewards_classic, rewards_fuel, rewards_wind, rewards_gravity, rewards_malfunction, title, window=50, save_path=None):
        Plots smoothed reward curves for multiple environments (Classic, Fuel, Wind, Gravity, and Malfunction) over episodes.
    
    plot_pid_dqn_comparison(dqn_rewards_classic, pid_rewards_classic, title, window=50, save_path=None):
        Compares the smoothed reward curves of DQN and PID controllers in the Classic environment.
    
    plot_performance_classic_and_gravity(rewards_classic, rewards_gravity, title, window=50, save_path=None):
        Plots and compares the smoothed reward curves of the Classic and Gravity environments over episodes.
    
    moving_average(data, window_size):
        Computes the moving average of the reward data with a specified window size to smooth the reward curves.
"""


def plot_performance(rewards_classic, rewards_fuel, rewards_wind, rewards_gravity, rewards_malfunction, title, window=50, save_path=None):

    smoothed_rewards_classic = moving_average(rewards_classic, window)
    smoothed_rewards_fuel = moving_average(rewards_fuel, window)
    smoothed_rewards_wind = moving_average(rewards_wind, window)
    smoothed_rewards_gravity = moving_average(rewards_gravity, window)
    smoothed_rewards_malfunction = moving_average(rewards_malfunction, window)

    plt.figure(figsize=(14, 6))

    # Smoothed Line Plot
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_fuel, label='Fuel Env', color='r')
    plt.plot(smoothed_rewards_wind, label='Wind Env', color='g')
    plt.plot(smoothed_rewards_gravity, label='Gravity Env', color='m')
    plt.plot(smoothed_rewards_malfunction, label='Malfunction Env', color='c') 
    
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title(title)
    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(save_path)

def plot_pid_dqn_comparison(dqn_rewards_classic, pid_rewards_classic, title, window=50, save_path=None):
    
    smoothed_dqn_classic = moving_average(dqn_rewards_classic, window)
    smoothed_pid_classic = moving_average(pid_rewards_classic, window)

    plt.figure(figsize=(14, 6))

    # Plot DQN rewards with a solid line
    plt.plot(smoothed_dqn_classic, label='DQN - Classic Environment', color='b', linestyle='-')

    # Plot PID rewards with a dotted dashed line
    plt.plot(smoothed_pid_classic, label='PID - Classic Environment', color='b', linestyle='--')

    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(save_path)

def plot_performance_classic_and_gravity(rewards_classic, rewards_gravity, title, window=50, save_path=None):

    smoothed_rewards_classic = moving_average(rewards_classic, window)
    smoothed_rewards_gravity = moving_average(rewards_gravity, window)

    plt.figure(figsize=(14, 6))

    # Smoothed Line Plot
    plt.plot(smoothed_rewards_classic, label='Classic Env', color='b')
    plt.plot(smoothed_rewards_gravity, label='Gravity Env', color='m')
    
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Reward')
    plt.title(title)
    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(save_path)
    
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')