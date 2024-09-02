import gymnasium as gym

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