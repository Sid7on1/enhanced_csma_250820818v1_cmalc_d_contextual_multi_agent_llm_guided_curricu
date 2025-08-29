import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
REPLAY_START_SIZE = 10000
REPLAY_FREQ = 4
UPDATE_FREQ = 100
GAMMA = 0.99
TAU = 0.001

# Enum for memory types
class MemoryType(Enum):
    EXPERIENCE = 1
    BUFFER = 2

# Dataclass for experience
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Abstract base class for memory
class Memory(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    @abstractmethod
    def add(self, experience: Experience):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Experience]:
        pass

# Class for experience replay memory
class ExperienceReplayMemory(Memory):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.memory = deque(maxlen=capacity)

    def add(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return np.random.choice(self.memory, batch_size, replace=False)

# Class for buffer memory
class BufferMemory(Memory):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.memory = deque(maxlen=capacity)
        self.index = 0
        self.buffer = np.zeros((capacity, 4))

    def add(self, experience: Experience):
        self.buffer[self.index] = np.array([experience.state, experience.action, experience.reward, experience.done])
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(self.capacity, batch_size, replace=False)
        return [Experience(self.buffer[i, 0], self.buffer[i, 1], self.buffer[i, 2], self.buffer[i, 3], False) for i in indices]

# Class for memory manager
class MemoryManager:
    def __init__(self, capacity: int, memory_type: MemoryType):
        self.capacity = capacity
        self.memory_type = memory_type
        if memory_type == MemoryType.EXPERIENCE:
            self.memory = ExperienceReplayMemory(capacity)
        elif memory_type == MemoryType.BUFFER:
            self.memory = BufferMemory(capacity)

    def add(self, experience: Experience):
        self.memory.add(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return self.memory.sample(batch_size)

# Class for experience replay
class ExperienceReplay:
    def __init__(self, capacity: int, memory_type: MemoryType):
        self.capacity = capacity
        self.memory_type = memory_type
        self.memory_manager = MemoryManager(capacity, memory_type)
        self.batch_size = BATCH_SIZE
        self.replay_start_size = REPLAY_START_SIZE
        self.replay_freq = REPLAY_FREQ
        self.update_freq = UPDATE_FREQ
        self.gamma = GAMMA
        self.tau = TAU

    def add(self, experience: Experience):
        self.memory_manager.add(experience)

    def sample(self) -> List[Experience]:
        return self.memory_manager.sample(self.batch_size)

    def update(self, model, target_model):
        experiences = self.sample()
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        # Compute Q-values
        q_values = model(states)
        next_q_values = target_model(next_states)

        # Compute TD-error
        td_errors = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones) - q_values[np.arange(len(experiences)), actions]

        # Update model
        model.update(states, actions, td_errors)

        # Update target model
        target_model.update(next_states, actions, td_errors)

# Class for memory
class Memory:
    def __init__(self):
        self.experience_replay = ExperienceReplay(MEMORY_CAPACITY, MemoryType.EXPERIENCE)

    def add(self, experience: Experience):
        self.experience_replay.add(experience)

    def sample(self) -> List[Experience]:
        return self.experience_replay.sample()

# Test the memory
if __name__ == "__main__":
    memory = Memory()
    experience = Experience(np.array([1, 2, 3]), 0, 1.0, np.array([4, 5, 6]), False)
    memory.add(experience)
    experiences = memory.sample()
    for experience in experiences:
        print(experience.state)