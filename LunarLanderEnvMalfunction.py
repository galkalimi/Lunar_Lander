import gymnasium as gym
import random


class LunarLanderEnvMalfunction(gym.Env):
    """
    A custom Gym environment wrapper for the LunarLander-v2 environment that simulates
    occasional engine malfunctions. The malfunction causes the agent's actions to be overridden,
    simulating a scenario where the engine may fail.

    Attributes:
        env (gym.Env): The base LunarLander-v2 environment.
        state (array): The current state of the environment.
        malfunction_probability (float): Probability of engine malfunction occurring at each step.
    """

    def __init__(self):
        """
        Initializes the LunarLanderEnvMalfunction environment.
        Sets up the base LunarLander-v2 environment and initializes malfunction parameters.
        """
        super(LunarLanderEnvMalfunction, self).__init__()
        self.env = gym.make('LunarLander-v2')  # Base environment
        self.state = None

        # Malfunction parameters
        self.malfunction_probability = 0.1  # 10% chance of engine malfunction

    def step(self, action):
        """
        Takes a step in the environment with the given action, possibly simulating a malfunction.

        Args:
            action (int): The action to be taken in the environment.

        Returns:
            tuple: A tuple containing:
                - next_state (array): The next state of the environment after taking the action.
                - reward (float): The reward received after taking the action.
                - done (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated.
                - info (dict): Additional information from the environment.
        """
        # Check if there is a malfunction
        if random.random() < self.malfunction_probability:
            # Malfunction occurs
            action = 0  # Force 'do nothing' action

        next_state, reward, done, truncated, info = self.env.step(action)
        self.state = next_state  # Update the state

        return next_state, reward, done, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment.

        Args:
            mode (str): The mode in which to render the environment. Defaults to 'human'.
        """
        self.env.render(mode)

    def close(self):
        """
        Closes the environment and performs any necessary cleanup.
        """
        self.env.close()

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            tuple: A tuple containing:
                - state (array): The initial state of the environment.
                - info (dict): Additional information from the environment.
        """
        self.state, info = self.env.reset()
        return self.state, info
