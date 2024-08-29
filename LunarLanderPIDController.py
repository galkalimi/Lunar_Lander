import gymnasium as gym
import time
from PIDController import PIDController

class LunarLanderPIDController:
    def __init__(self, env):
        self.env = env
        self.vertical_pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.5, setpoint=0)  # Adjust parameters
        self.horizontal_pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.5, setpoint=0)  # Adjust parameters
        self.angle_pid = PIDController(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=0)  # Adjust parameters

    def select_action(self, state):
        x, y, vx, vy, theta, omega, left_leg_contact, right_leg_contact = state

        # Vertical control
        thrust = self.vertical_pid.compute(vy, dt=1/50)  # Assuming 50Hz update rate

        # Horizontal control
        side_thrust = self.horizontal_pid.compute(vx, dt=1/50)

        # Angular control
        angle_adjustment = self.angle_pid.compute(theta, dt=1/50)

        # Convert PID output to discrete actions
        if thrust > 0.6:
            action = 2  # Main engine fire
        elif side_thrust > 0.1:
            action = 3  # Fire right engine
        elif side_thrust < -0.1:
            action = 1  # Fire left engine
        else:
            action = 0  # Do nothing

        return action

    def run(self):
        for i in range(25):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, truncated, info = self.env.step(action)
                self.env.render()
                time.sleep(0.02)  # Slow down rendering to make it visible

        self.env.close()