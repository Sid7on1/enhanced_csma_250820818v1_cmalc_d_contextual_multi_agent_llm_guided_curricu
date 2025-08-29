import os
import logging
from typing import Dict, List, Tuple, Union
import numpy as np
from numpy.random import default_rng

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    """
    Configuration class for the agent and environment.

    Parameters:
    -----------
    env_name: str
        Name of the environment.
    num_agents: int
        Number of agents in the environment.
    max_episodes: int
        Maximum number of episodes to run.
    max_steps: int
        Maximum number of steps per episode.
    seed: int, optional
        Random seed for reproducibility.
    """
    def __init__(self, env_name: str, num_agents: int, max_episodes: int, max_steps: int, seed: int = None):
        self.env_name = env_name
        self.num_agents = num_agents
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.seed = seed

        # Set random seed for reproducibility
        if self.seed:
            self.set_seed(self.seed)

        # Environment-specific configurations
        if env_name == 'custom_env':
            self.env_config = CustomEnvConfig()
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

        # Agent-specific configurations
        self.agent_config = AgentConfig()

    def set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.

        Parameters:
        ----------
        seed: int
            Random seed value.
        """
        np.random.seed(seed)

    # Add more configuration methods as needed

# Environment-specific configuration class
class CustomEnvConfig:
    """
    Configuration class for the custom environment.

    Attributes:
    -----------
    observation_space: Dict[str, Union[int, float]]
        Dictionary defining the observation space.
    action_space: Dict[str, Union[int, float]]
        Dictionary defining the action space.
    reward_threshold: float
        Minimum reward required to consider the task 'solved'.
    """
    def __init__(self):
        self.observation_space = {
            'shape': (10,),
            'low': -10.0,
            'high': 10.0
        }
        self.action_space = {
            'n': 5,
            'low': -1.0,
            'high': 1.0
        }
        self.reward_threshold = 15.0  # Task is considered solved if average reward > reward_threshold

# Agent-specific configuration class
class AgentConfig:
    """
    Configuration class for the agent(s).

    Attributes:
    -----------
    algorithm: str
        Name of the algorithm used by the agent(s).
    learning_rate: float
        Learning rate for the agent's update rule.
    gamma: float
        Discount factor for future rewards.
    tau: float
        Soft update coefficient for the target network.
    buffer_size: int
        Maximum number of experiences to store in the replay buffer.
    batch_size: int
        Number of experiences to sample per training update.
    """
    def __init__(self):
        self.algorithm = 'MARL-cMALC-D'  # Multi-Agent Reinforcement Learning with Contextual Curriculum Learning and Diversity
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.tau = 0.01
        self.buffer_size = 10000
        self.batch_size = 64

# Function to create configurations from a file
def create_config_from_file(config_file: str) -> Config:
    """
    Create a Config object from a configuration file.

    Parameters:
    -----------
    config_file: str
        Path to the configuration file.

    Returns:
    --------
    config: Config
        Configuration object.

    Raises:
    --------
    FileNotFoundError: If the configuration file is not found.
    ValueError: If the configuration file is missing required parameters.
    """
    config_path = os.path.abspath(os.path.expanduser(config_file))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_path, 'r') as file:
            config_data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding configuration file: {e}")

    # Extract required parameters
    env_name = config_data.get('env_name')
    num_agents = config_data.get('num_agents')
    max_episodes = config_data.get('max_episodes')
    max_steps = config_data.get('max_steps')
    seed = config_data.get('seed', None)

    # Validate parameters
    if not env_name:
        raise ValueError("Missing 'env_name' parameter in configuration file")
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError("Invalid 'num_agents' parameter in configuration file")
    if not isinstance(max_episodes, int) or max_episodes <= 0:
        raise ValueError("Invalid 'max_episodes' parameter in configuration file")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("Invalid 'max_steps' parameter in configuration file")

    # Create the Config object
    config = Config(env_name, num_agents, max_episodes, max_steps, seed)

    # Add optional parameters to the config object
    # ...

    return config

# Function to create configurations from command-line arguments
def create_config_from_cli(args: List[str]) -> Config:
    """
    Create a Config object from command-line arguments.

    Parameters:
    -----------
    args: List[str]
        List of command-line arguments.

    Returns:
    --------
    config: Config
        Configuration object.

    Raises:
    --------
    ValueError: If required command-line arguments are missing.
    """
    if len(args) < 5:
        raise ValueError("Missing command-line arguments. Usage: python <file_name> <env_name> <num_agents> <max_episodes> <max_steps> [seed]")

    env_name = args[1]
    num_agents = int(args[2])
    max_episodes = int(args[3])
    max_steps = int(args[4])
    seed = int(args[5]) if len(args) > 5 else None

    # Create the Config object
    config = Config(env_name, num_agents, max_episodes, max_steps, seed)

    return config

# Function to validate configurations
def validate_config(config: Config) -> None:
    """
    Validate the Config object.

    Parameters:
    -----------
    config: Config
        Configuration object to validate.

    Raises:
    --------
    ValueError: If any validation checks fail.
    """
    # Validate environment name
    if not config.env_name:
        raise ValueError("Environment name cannot be empty")

    # Validate number of agents
    if config.num_agents <= 0:
        raise ValueError("Number of agents must be greater than 0")

    # Validate maximum episodes and steps
    if config.max_episodes <= 0:
        raise ValueError("Maximum number of episodes must be greater than 0")
    if config.max_steps <= 0:
        raise ValueError("Maximum number of steps per episode must be greater than 0")

    # Add more validation checks as needed

# Main function to create configurations
def create_config(config_file: str = None, args: List[str] = None) -> Config:
    """
    Create a Config object from either a configuration file or command-line arguments.

    Parameters:
    -----------
    config_file: str, optional
        Path to the configuration file.
    args: List[str], optional
        List of command-line arguments.

    Returns:
    --------
    config: Config
        Configuration object.

    Raises:
    --------
    ValueError: If both config_file and args are provided, or if neither is provided.
    """
    if (config_file is None and args is None) or (config_file is not None and args is not None):
        raise ValueError("Provide either a configuration file or command-line arguments, but not both.")

    if config_file:
        config = create_config_from_file(config_file)
    else:
        config = create_config_from_cli(args)

    validate_config(config)

    return config

# Example usage
if __name__ == '__main__':
    # Example configuration file: config.json
    # {
    #     "env_name": "custom_env",
    #     "num_agents": 3,
    #     "max_episodes": 1000,
    #     "max_steps": 200,
    #     "seed": 42
    # }

    # Create config from file
    # config = create_config(config_file='config.json')

    # Create config from command-line arguments
    # config = create_config(args=['python', 'config.py', 'custom_env', '3', '1000', '200', '42'])

    # Print configuration summary
    # logger.info(config)

    # Access specific configuration values
    # env_name = config.env_name
    # num_agents = config.num_agents
    # ...