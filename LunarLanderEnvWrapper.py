import gymnasium as gym

#LunarLanderEnvWrapper
class LunarLanderEnvWrapper(gym.Env):
    def __init__(self, gravity=(0,-10), enable_wind=False, wind_power=15.0):
        super(LunarLanderEnvWrapper, self).__init__()
        self.env = gym.make('LunarLander-v2', enable_wind=enable_wind, wind_power=wind_power)  # Base environment
        self.state = None
        
        # Set custom gravity
        self.gravity = gravity
        self.env.unwrapped.world.gravity = self.gravity
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        self.state, info = self.env.reset()
        return self.state, info

    def step(self, action):
        # Perform the action in the environment
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()