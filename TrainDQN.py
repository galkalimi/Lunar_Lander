import torch

def train_dqn(env, agent, model_name, num_episodes=4200, update_target_every=10, max_steps_per_episode=200):
    """
    Trains a DQN agent in a specified environment using the given parameters.

    Args:
        env (gym.Env): The environment to train the agent in.
        agent (DQNAgent): The agent to train, including the policy and target networks.
        model_name (str): The name for saving the trained model.
        num_episodes (int, optional): The total number of training episodes. Defaults to 4200.
        update_target_every (int, optional): Frequency (in episodes) of updating the target network. Defaults to 10.
        max_steps_per_episode (int, optional): Maximum steps to run per episode. Defaults to 200.

    The function performs the following steps:
    - Resets the environment and starts an episode.
    - At each step, the agent selects an action, performs it, stores the transition in memory, and optimizes the model.
    - Every `update_target_every` episodes, the target network is updated to match the policy network.
    - The trained model is saved to disk at the end of training, and the environment is closed.
    """
    
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

    agent.save_model(model_name + '.pth')
    env.close()
