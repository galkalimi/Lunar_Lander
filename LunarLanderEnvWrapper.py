import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium import spaces


#LunarLanderEnvWrapper
class LunarLanderEnvWrapper(gym.Env):
    def __init__(self, gravity=(0,-10), enable_wind=False, wind_power=15.0):
        super(LunarLanderEnvWrapper, self).__init__()
        self.env = gym.make('LunarLander-v2', render_mode="human", enable_wind=enable_wind, wind_power=wind_power)  # Base environment
        self.state = None
        
        # Set custom gravity
        self.gravity = gravity
        self.env.unwrapped.world.gravity = self.gravity

        # Fuel parameters - custom modification
        self.fuel_limit = 1000  # Maximum fuel amount
        self.current_fuel = self.fuel_limit
        self.prev_fuel = self.fuel_limit
        self.fuel_consumption_rate = 10  # Amount of fuel consumed per thrust action

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
        # Adjust action based on fuel
        if action in [1, 2, 3] and self.current_fuel >= self.fuel_consumption_rate:
            self.current_fuel -= self.fuel_consumption_rate
        else:
            action = 0  # If no fuel, force 'do nothing' action

        # Perform the action in the environment
        next_state, reward, done, truncated, info = self.env.step(action)

        # Append fuel level to the state
        next_state_with_fuel = np.append(next_state, self.current_fuel)

        # Check if the episode should terminate due to fuel depletion
        if self.current_fuel <= 0:
            print("Ran out of fuel")
            done = True
            reward -= 50  # Penalize for running out of fuel
            info['termination_reason'] = 'out_of_fuel'  # Add reason to info

        # Custom termination condition based on landing
        if self.state[6]:
            done = True
            print("touched the ground")

        return next_state_with_fuel, reward, done, truncated, info


    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()