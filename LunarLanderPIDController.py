import gymnasium as gym
import time
from PIDController import PIDController

class LunarLanderPIDController:
    def __init__(self, env):
        self.env = env
        self.vertical_pid = PIDController(Kp=0.2, Ki=0.0, Kd=0.7, setpoint=0)  
        self.horizontal_pid = PIDController(Kp=0.2, Ki=0.0, Kd=0.7, setpoint=0)  
        self.angle_pid = PIDController(Kp=0.2, Ki=0.0, Kd=0.6, setpoint=0) 

    def select_action(self, state):
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

    def run(self):
        total_scores = []
        for i in range(5000):  # Run multiple episodes
            state, _ = self.env.reset()
            done = False
            episode_reward = 0  # Track total reward for the episode
            while not done:
                action = self.select_action(state)
                state, reward, done, truncated, info = self.env.step(action)
                self.env.render()
                # time.sleep(0.02)  # Slow down rendering to make it visible
                episode_reward += reward  # Accumulate reward

            total_scores.append(episode_reward)  # Store the total reward for this episode
            print(f'Episode {i+1} Total Reward: {episode_reward}')
    
        self.env.close()
        average_score = sum(total_scores) / len(total_scores)
        print(f'Average Reward over {len(total_scores)} episodes: {average_score}')