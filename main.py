from re import T
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gymnasium import spaces
from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from LunarLanderEnvWrapper import LunarLanderEnvWrapper
from DQN import DQN

#main
def make_gravity_rewards_comparison():
    # Example data
    gravities = np.array([-1000, -1500, -2000, -2500])
    average_scores = np.array([120, 90, 60, 30])

    # Dark theme settings
    plt.style.use('dark_background')
    colors = plt.cm.cividis(np.linspace(0.2, 0.8, len(gravities)))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(gravities, average_scores, color=colors, width=200)

    # Adding titles and labels
    plt.title('Average Score vs. Gravity', fontsize=18, color='white')
    plt.xlabel('Gravity', fontsize=14, color='white')
    plt.ylabel('Average Score', fontsize=14, color='white')

    # Customize tick labels
    plt.xticks(gravities, fontsize=12, color='white')
    plt.yticks(fontsize=12, color='white')

    # Display the graph
    plt.show()


# Training function
def train_dqn(env, agent, num_episodes=5000, update_target_every=10, max_steps_per_episode=200):
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            agent.memory.push((torch.tensor(state, dtype=torch.float32),
                               torch.tensor(action),
                               torch.tensor(reward, dtype=torch.float32),
                               torch.tensor(next_state, dtype=torch.float32),
                               torch.tensor(done, dtype=torch.float32)))

            agent.optimize_model()
            state = next_state

            if done or truncated:
                break

        if episode % update_target_every == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    # Save the trained model
    agent.save_model('dqn_lunarlander.pth')
    env.close()


def test_dqn(env, agent, num_episodes=100, max_steps_per_episode=200):
    avg_reward = 0
    agent.epsilon = 0.0  # Disable exploration for testing
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        print(f'Test Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')
        avg_reward += total_reward

    avg_reward /= num_episodes
    print(f'Average Test Reward: {avg_reward}')
    env.close()




# Main function
def main():
    # render_mode="human" (add to env)
    env = LunarLanderEnvWrapper(gravity=(0,-10), enable_wind=True, wind_power=2000.0)  # Using custom environment with fuel
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_space = env.action_space
    agent = DQNAgent(state_dim, action_dim, action_space)
    
    # Train the agent
    print("Training the agent...")
    train_dqn(env, agent)

    # Load the trained model
    agent.load_model('dqn_lunarlander.pth')

    # Test the agent
    print("Testing the trained agent...")
    test_env = LunarLanderEnvWrapper(gravity=(0,-100), enable_wind=True, wind_power=2000.0)  # Using custom environment with fuel
    test_dqn(test_env, agent)

if __name__ == "__main__":
    main()