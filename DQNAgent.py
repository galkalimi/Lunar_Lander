import torch
import torch.nn as nn
import torch.optim as optim
import random
from ReplayBuffer import ReplayBuffer
from DQN import DQN

class DQNAgent:
    """
    A Deep Q-Network (DQN) agent for reinforcement learning tasks.

    This class implements a DQN agent that interacts with an environment to learn optimal actions
    using experience replay and target networks. It includes methods for action selection, model 
    optimization, and saving/loading the model.

    Attributes:
        state_dim (int): The dimensionality of the input state space.
        action_dim (int): The dimensionality of the action space.
        action_space (gym.Space): The action space of the environment.
        batch_size (int): The number of transitions to sample from the replay buffer for training.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The initial exploration rate.
        epsilon_min (float): The minimum exploration rate.
        epsilon_decay (float): The decay rate of epsilon.
        lr (float): The learning rate for the optimizer.
        memory_capacity (int): The maximum number of transitions to store in the replay buffer.
        dropout_prob (float): The dropout probability used in the DQN model.

    Methods:
        __init__(self, state_dim, action_dim, action_space, batch_size=64, gamma=0.99, epsilon=1.0, 
                epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, memory_capacity=10000, dropout_prob=0):
            Initializes the DQN agent, including the policy and target networks, optimizer, and replay buffer.

        select_action(self, state):
            Selects an action based on the current policy or randomly with probability epsilon.
        
        optimize_model(self):
            Samples a batch of transitions from the replay buffer and updates the policy network by minimizing 
            the loss between predicted and target Q-values.

        save_model(self, path):
            Saves the policy network's state dictionary to the specified path.

        load_model(self, path):
            Loads the policy network's state dictionary from the specified path and synchronizes the target network.

        run(self, env, num_iterations, stop_event):
            Executes the agent in the environment for a given number of iterations, taking actions and updating 
            the agent's knowledge.
    """

    def __init__(self, state_dim, action_dim, action_space, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, lr=0.001, memory_capacity=10000):
        """
          Initializes the DQN agent with the specified parameters.

          Args:
              state_dim (int): The dimensionality of the state space.
              action_dim (int): The dimensionality of the action space.
              action_space (gym.Space): The action space of the environment.
              batch_size (int, optional): Number of transitions to sample from replay buffer. Defaults to 64.
              gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
              epsilon (float, optional): Initial exploration rate. Defaults to 1.0.
              epsilon_min (float, optional): Minimum exploration rate. Defaults to 0.01.
              epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.995.
              lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
              memory_capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
              dropout_prob (float, optional): Dropout probability in the DQN model. Defaults to 0.
          """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.memory = ReplayBuffer(memory_capacity)
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The action selected by the policy or randomly with probability epsilon.
        """
        if random.random() < self.epsilon:
            action = self.action_space.sample()  # Choose a random action
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                action = self.policy_net(state).argmax().item()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action

    def optimize_model(self):
        """
        Optimizes the policy network by sampling a batch of transitions from the replay buffer and 
        minimizing the loss between predicted and target Q-values.
        """
        if len(self.memory) < self.batch_size:
            return

            # Sample a batch of transitions from the replay buffer
            transitions = self.memory.sample(self.batch_size)

            # Unpack the batch into separate tensors
            batch = list(zip(*transitions))
            states = torch.stack(batch[0])
            actions = torch.stack(batch[1])
            rewards = torch.stack(batch[2])
            next_states = torch.stack(batch[3])
            dones = torch.stack(batch[4])

            # Compute the predicted Q-values for the current states
            state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute the target Q-values for the next states using the target network
            next_state_values = self.target_net(next_states).max(1)[0]

            # Compute the expected Q-values for the current states
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

            # Compute the loss between the predicted Q-values and the expected Q-values
            loss = self.criterion(state_action_values, expected_state_action_values.detach())

            # Perform gradient descent to update the policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, path):
        """
        Saves the policy network's state dictionary to the specified file path.

        Args:
            path (str): The file path where the model will be saved.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """
        Loads the policy network's state dictionary from the specified file path and synchronizes 
        the target network with the policy network.

        Args:
            path (str): The file path from which to load the model.
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run(self, env, num_episodes=100, max_steps_per_episode=200):
        """
        Runs the agent in the environment for a specified number of iterations, performing actions and 
        rendering the environment.

        Args:
            env (gym.Env): The environment in which the agent operates.
            num_episodes (int): The number of episodes to run.
            max_steps_per_episode (int): The maximum number of steps per episode
        """
        avg_reward = 0
        self.epsilon = 0.0  # Disable exploration for testing
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                state = next_state

                if done or truncated:
                    break

            avg_reward += total_reward

        env.close()