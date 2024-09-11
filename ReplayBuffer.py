# Description: Replay buffer for DQN agent to store transitions and sample mini-batches for training.

from collections import deque
import random

class ReplayBuffer:
    """
    A Replay Buffer for storing and sampling transitions, typically used in reinforcement learning.

    The ReplayBuffer stores transitions up to a specified capacity. It allows adding new transitions
    and sampling a random batch of transitions for training purposes.

    Attributes:
        buffer (deque): The internal buffer storing the transitions.

    Methods:
        push(transition):
            Adds a transition to the buffer.
        
        sample(batch_size):
            Samples a random batch of transitions from the buffer.
        
        __len__():
            Returns the current number of transitions stored in the buffer.

    Example:
        buffer = ReplayBuffer(capacity=10000)
        buffer.push(transition)
        batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity):
        """
        Initializes the ReplayBuffer with a given capacity.

        Args:
            capacity (int): The maximum number of transitions the buffer can hold.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """
        Adds a transition to the buffer. If the buffer is full, the oldest transition is removed.

        Args:
            transition (Any): The transition to be added to the buffer.
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list containing the sampled transitions.

        Raises:
            ValueError: If the batch_size is greater than the number of transitions stored.
        """
        if batch_size > len(self.buffer):
            raise ValueError("Sample size greater than number of elements in buffer")
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Returns the current number of transitions stored in the buffer.

        Returns:
            int: The number of transitions in the buffer.
        """
        return len(self.buffer)

