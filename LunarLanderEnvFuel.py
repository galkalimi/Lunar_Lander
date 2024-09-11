import gymnasium as gym

class LunarLanderEnvFuel(gym.Env):
    """
    A custom wrapper for the Lunar Lander environment that adds fuel consumption mechanics.

    This environment extends the standard 'LunarLander-v2' environment from Gymnasium by 
    introducing a fuel system. The agent has a limited amount of fuel, and thrust actions 
    consume fuel. If the fuel is depleted, the agent can only perform the 'do nothing' action.

    Attributes:
        env (gym.Env): The base 'LunarLander-v2' environment.
        state (numpy.ndarray): The current state of the environment.
        fuel_limit (int): The maximum amount of fuel available at the start.
        current_fuel (int): The amount of fuel currently available.
        prev_fuel (int): The amount of fuel before the last action.
        fuel_consumption_rate (int): The amount of fuel consumed per thrust action.
        action_space (gym.spaces.Space): The action space of the environment.

    Methods:
        step(action):
            Executes the given action in the environment, considering fuel constraints.
        render(mode='human'):
            Renders the environment.
        close():
            Closes the environment.
        reset():
            Resets the environment and refills the fuel.
    
    Example:
        env = LunarLanderEnvFuel()
        state, info = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        env.render()
        env.close()
    """

    def __init__(self):
        """
        Initializes the LunarLanderEnvFuel with the base environment and fuel parameters.
        """
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
        """
        Executes the given action in the environment, considering fuel constraints.

        Args:
            action (int): The action to perform.

        Returns:
            tuple: A tuple containing the next state, reward, done flag, truncated flag, and info dictionary.
        """
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
        """
        Renders the environment.

        Args:
            mode (str): The mode to render with. Default is 'human'.
        """
        self.env.render(mode)

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()

    def reset(self):
        """
        Resets the environment and refills the fuel.

        Returns:
            tuple: A tuple containing the initial state and an info dictionary.
        """
        self.state, info = self.env.reset()
        self.current_fuel = self.fuel_limit
        return self.state, info
