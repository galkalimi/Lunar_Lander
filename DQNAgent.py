import torch
import torch.nn as nn
import torch.optim as optim
import random
from ReplayBuffer import ReplayBuffer
from DQN import DQN

class DQNAgent:
    """
    A Deep Q-Network (DQN) agent for reinforcement learning tasks.

    This agent interacts with an environment by selecting actions based on an 
    epsilon-greedy policy. It uses a replay buffer to store transitions and 
    performs optimization by training a policy network to minimize the difference 
    between predicted Q-values and target Q-values. The agent also maintains a target 
    network for stable learning.

    Attributes:
        state_dim (int): The dimensionality of the input state space.
        action_dim (int): The dimensionality of the output action space.
        action_space (gym.spaces.Space): The action space of the environment.
        batch_size (int): The size of the batch used for training.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The initial exploration rate for the epsilon-greedy policy.
        epsilon_min (float): The minimum exploration rate.
        epsilon_decay (float): The decay rate of epsilon after each action.
        lr (float): The learning rate for the optimizer.
        memory (ReplayBuffer): The replay buffer to store transitions.
        policy_net (DQN): The policy network used to select actions.
        target_net (DQN): The target network used to calculate target Q-values.
        optimizer (torch.optim.Optimizer): The optimizer for training the policy network.
        criterion (torch.nn.Module): The loss function used to calculate the training loss.

    Methods:
        select_action(state):
            Selects an action based on the current state using an epsilon-greedy policy.
        optimize_model():
            Samples a batch from the replay buffer and optimizes the policy network.
        save_model(path):
            Saves the policy network's weights to a specified file path.
        load_model(path):
            Loads the policy network's weights from a specified file path and updates the target network.
    
    Example:
        agent = DQNAgent(state_dim=8, action_dim=4, action_space=env.action_space)
        action = agent.select_action(state)
        agent.optimize_model()
        agent.save_model("dqn_model.pth")
        agent.load_model("dqn_model.pth")
    """

    def __init__(self, state_dim, action_dim, action_space, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, lr=0.001, memory_capacity=10000):
        """
        Initializes the DQNAgent with the given parameters and sets up the policy and target networks.

        Args:
            state_dim (int): The dimensionality of the input state space.
            action_dim (int): The dimensionality of the output action space.
            action_space (gym.spaces.Space): The action space of the environment.
            batch_size (int, optional): The size of the batch used for training. Defaults to 64.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.
            epsilon (float, optional): The initial exploration rate for the epsilon-greedy policy. Defaults to 1.0.
            epsilon_min (float, optional): The minimum exploration rate. Defaults to 0.01.
            epsilon_decay (float, optional): The decay rate of epsilon after each action. Defaults to 0.995.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
            memory_capacity (int, optional): The maximum size of the replay buffer. Defaults to 10000.
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
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            int: The selected action.
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
        Samples a batch from the replay buffer and optimizes the policy network.

        Returns:
            None
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
        Saves the policy network's weights to a specified file path.

        Args:
            path (str): The file path where the model weights will be saved.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """
        Loads the policy network's weights from a specified file path and updates the target network.

        Args:
            path (str): The file path from which the model weights will be loaded.
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
