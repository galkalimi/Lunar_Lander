import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) implementation using PyTorch.

    This neural network model is designed for reinforcement learning tasks, where 
    it approximates the Q-value function. The architecture consists of fully 
    connected layers with ReLU activations, and it is tailored for environments 
    with a given state and action dimensionality.

    Attributes:
        fc1 (torch.nn.Linear): The first fully connected layer with 256 neurons.
        fc2 (torch.nn.Linear): The second fully connected layer with 256 neurons.
        fc3 (torch.nn.Linear): An additional fully connected layer with 128 neurons.
        fc4 (torch.nn.Linear): The output layer that maps to the action space.

    Methods:
        forward(x):
            Defines the forward pass of the network.
    
    Example:
        model = DQN(state_dim=8, action_dim=4)
        output = model(torch.tensor([state], dtype=torch.float32))
    """

    def __init__(self, state_dim, action_dim):
        """
        Initializes the DQN model with specified input (state) and output (action) dimensions.

        Args:
            state_dim (int): The dimensionality of the input state space.
            action_dim (int): The dimensionality of the output action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # Increased number of neurons
        self.fc2 = nn.Linear(256, 256)  # Increased number of neurons
        self.fc3 = nn.Linear(256, 128)  # Additional layer
        self.fc4 = nn.Linear(128, action_dim)  # Additional layer

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
