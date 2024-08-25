import gymnasium
import torch
import gymnasium as gym

import LunarLanderEnvWrapper
from DQNAgent import DQNAgent


def train_dqn(env, agent, num_episodes=1000, update_target_every=10, max_steps_per_episode=200):
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
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


def test_dqn(env, agent, num_episodes=10, max_steps_per_episode=200):
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

    env.close()


# Main function
def main():
    env = LunarLanderEnvWrapper()  # Using custom environment with fuel
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_space = env.action_space
    agent = DQNAgent(state_dim, action_dim, action_space)

    # Train the agent
    # print("Training the agent...")
    # train_dqn(env, agent)

    # Load the trained model
    agent.load_model('dqn_lunarlander.pth')

    # Test the agent
    print("Testing the trained agent...")
    test_dqn(env, agent)

if __name__ == "__main__":
    main()