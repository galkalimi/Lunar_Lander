import gymnasium as gym
import numpy as np
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