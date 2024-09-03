import gymnasium as gym

class LunarLanderEnvWrapper(gym.Env):
    """
    A custom wrapper for the Lunar Lander environment from Gymnasium.

    This wrapper allows for custom gravity settings and optional wind conditions 
    to be applied to the Lunar Lander environment. It provides methods to reset 
    the environment, perform actions, render the environment, and close it.

    Attributes:
        env (gym.Env): The base Lunar Lander environment from Gymnasium.
        state (numpy.ndarray): The current state of the environment.
        gravity (tuple): The gravity setting for the environment, applied as (x, y) vector.
        action_space (gym.spaces.Space): The action space of the environment.
        observation_space (gym.spaces.Space): The observation space of the environment.

    Methods:
        reset():
            Resets the environment to its initial state and returns the initial observation.
        step(action):
            Takes an action in the environment and returns the next state, reward, 
            done flag, truncated flag, and additional info.
        render(mode='human'):
            Renders the environment.
        close():
            Closes the environment and releases any resources.
    
    Example:
        env = LunarLanderEnvWrapper(gravity=(0, -9.8), enable_wind=True, wind_power=10.0)
        state, info = env.reset()
        next_state, reward, done, truncated, info = env.step(action)
    """

    def __init__(self, gravity=(0,-10), enable_wind=False, wind_power=15.0):
        """
        Initializes the LunarLanderEnvWrapper with custom gravity and wind settings.

        Args:
            gravity (tuple, optional): A tuple representing gravity as an (x, y) vector. Defaults to (0, -10).
            enable_wind (bool, optional): Whether to enable wind in the environment. Defaults to False.
            wind_power (float, optional): The strength of the wind if enabled. Defaults to 15.0.
        """
        super(LunarLanderEnvWrapper, self).__init__()
        self.env = gym.make('LunarLander-v2', enable_wind=enable_wind, wind_power=wind_power)  # Base environment
        self.state = None
        
        # Set custom gravity
        self.gravity = gravity
        self.env.unwrapped.world.gravity = self.gravity
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            tuple: The initial state and additional information.
        """
        self.state, info = self.env.reset()
        return self.state, info

    def step(self, action):
        """
        Performs an action in the environment.

        Args:
            action (int): The action to be taken.

        Returns:
            tuple: The next state, reward, done flag, truncated flag, and additional info.
        """
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment.

        Args:
            mode (str, optional): The mode in which to render. Defaults to 'human'.
        
        Returns:
            The rendered frame, depending on the mode.
        """
        return self.env.render()

    def close(self):
        """
        Closes the environment and releases any resources.
        """
        self.env.close()
