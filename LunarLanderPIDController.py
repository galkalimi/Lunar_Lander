
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from PIDController import PIDController

class LunarLanderPIDController:
    """
    A controller for the Lunar Lander environment using PID controllers for vertical, horizontal, 
    and angular control.

    This class implements a PID-based control strategy for navigating the Lunar Lander. It uses 
    three separate PID controllers to manage the vertical velocity, horizontal velocity, and angle 
    of the lander. Based on the output of these controllers, the appropriate action is selected 
    to control the lander.

    Attributes:
        env (gym.Env): The Lunar Lander environment instance.
        vertical_pid (PIDController): PID controller for vertical velocity.
        horizontal_pid (PIDController): PID controller for horizontal velocity.
        angle_pid (PIDController): PID controller for angular position.

    Methods:
        select_action(state):
            Determines the appropriate action based on the current state using the PID controllers.
        run():
            Runs multiple episodes of the environment with the PID controller and tracks performance.
    
    Example:
        env = LunarLanderEnvFuel()
        controller = LunarLanderPIDController(env)
        controller.run()
    """

    def __init__(self, env):
        """
        Initializes the LunarLanderPIDController with the given environment and sets up 
        the PID controllers for vertical, horizontal, and angular control.

        Args:
            env (gym.Env): The Lunar Lander environment instance.
        """
        self.env = env
        self.vertical_pid = PIDController(Kp=0.2, Ki=0.0, Kd=0.7, setpoint=0)  
        self.horizontal_pid = PIDController(Kp=0.2, Ki=0.0, Kd=0.7, setpoint=0)  
        self.angle_pid = PIDController(Kp=0.2, Ki=0.0, Kd=0.6, setpoint=0) 

    def select_action(self, state):
        """
        Determines the appropriate action based on the current state using the PID controllers.

        Args:
            state (tuple): The current state of the environment, which includes position, 
                           velocity, angle, and leg contact information.

        Returns:
            int: The selected action (0: do nothing, 1: fire left engine, 2: fire main engine, 
                 3: fire right engine).
        """
        x, y, vx, vy, theta, omega, left_leg_contact, right_leg_contact = state

        # Vertical control
        thrust = self.vertical_pid.compute(vy, dt=1/25)  # Assuming 25Hz update rate

        # Horizontal control
        side_thrust = self.horizontal_pid.compute(vx, dt=1/25)

        # Angular control
        angle_adjustment = self.angle_pid.compute(theta, dt=1/25)

        # Convert PID output to discrete actions
        if thrust > 0.5:
            action = 2  # Main engine fire
        elif side_thrust > 0.1:
            action = 3  # Fire right engine
        elif side_thrust < -0.1:
            action = 1  # Fire left engine
        else:
            action = 0  # Do nothing

        return action

    def run(self, stop_event=None, num_iterations=100):
        """
        Runs multiple episodes of the environment with the PID controller and tracks performance.

        This method resets the environment and runs through multiple episodes, using the 
        `select_action` method to determine the actions at each timestep. The total reward 
        for each episode is recorded and an average reward is calculated at the end.

        Returns:
            None
        """
        for episode in range(num_iterations):  # Run multiple episodes
            state, _ = self.env.reset()
            done = False
            episode_reward = 0  # Track total reward for the episode
            episode_reward = 0  # Track total reward for the episode
            while not done:
                action = self.select_action(state)
                state, reward, done, truncated, info = self.env.step(action)
                self.env.render()

                 # Collecting error data for plotting
                x, y, vx, vy, theta, omega, left_leg_contact, right_leg_contact = state
                error_vy = self.vertical_pid.setpoint - vy
                error_vx = self.horizontal_pid.setpoint - vx
                error_theta = self.angle_pid.setpoint - theta
                   

                 # Collecting error data for plotting
                x, y, vx, vy, theta, omega, left_leg_contact, right_leg_contact = state
                error_vy = self.vertical_pid.setpoint - vy
                error_vx = self.horizontal_pid.setpoint - vx
                error_theta = self.angle_pid.setpoint - theta
                   
        self.env.close()