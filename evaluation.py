import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationException(Exception):
    """Base class for evaluation exceptions."""
    pass

class InvalidMetricException(EvaluationException):
    """Raised when an invalid metric is specified."""
    pass

class EvaluationConfig:
    """Configuration for evaluation."""
    def __init__(self, 
                 metric: str = 'mse', 
                 threshold: float = 0.5, 
                 batch_size: int = 32, 
                 num_workers: int = 4):
        """
        Initialize evaluation configuration.

        Args:
        - metric (str): Evaluation metric. Defaults to 'mse'.
        - threshold (float): Threshold for evaluation. Defaults to 0.5.
        - batch_size (int): Batch size for evaluation. Defaults to 32.
        - num_workers (int): Number of workers for evaluation. Defaults to 4.
        """
        self.metric = metric
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

class EvaluationDataset(Dataset):
    """Dataset for evaluation."""
    def __init__(self, 
                 data: pd.DataFrame, 
                 labels: pd.Series, 
                 scaler: StandardScaler):
        """
        Initialize evaluation dataset.

        Args:
        - data (pd.DataFrame): Data for evaluation.
        - labels (pd.Series): Labels for evaluation.
        - scaler (StandardScaler): Scaler for data.
        """
        self.data = scaler.fit_transform(data)
        self.labels = labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get item from dataset.

        Args:
        - index (int): Index of item.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Item and label.
        """
        return self.data[index], self.labels[index]

class Evaluator:
    """Evaluator for agent."""
    def __init__(self, 
                 config: EvaluationConfig, 
                 model: torch.nn.Module, 
                 device: torch.device):
        """
        Initialize evaluator.

        Args:
        - config (EvaluationConfig): Evaluation configuration.
        - model (torch.nn.Module): Model for evaluation.
        - device (torch.device): Device for evaluation.
        """
        self.config = config
        self.model = model
        self.device = device

    def evaluate(self, 
                  data: pd.DataFrame, 
                  labels: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
        - data (pd.DataFrame): Data for evaluation.
        - labels (pd.Series): Labels for evaluation.

        Returns:
        - Dict[str, float]: Evaluation metrics.
        """
        try:
            # Create dataset and data loader
            scaler = StandardScaler()
            dataset = EvaluationDataset(data, labels, scaler)
            data_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

            # Initialize metrics
            metrics = {}

            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                total_loss = 0
                total_mse = 0
                total_mae = 0
                for batch in data_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = torch.nn.MSELoss()(outputs, targets)
                    total_loss += loss.item()
                    total_mse += mean_squared_error(targets.cpu().numpy(), outputs.cpu().numpy())
                    total_mae += mean_absolute_error(targets.cpu().numpy(), outputs.cpu().numpy())

            # Calculate metrics
            metrics['loss'] = total_loss / len(data_loader)
            metrics['mse'] = total_mse / len(data_loader)
            metrics['mae'] = total_mae / len(data_loader)

            # Log metrics
            logger.info(f'Evaluation metrics: {metrics}')

            return metrics

        except Exception as e:
            logger.error(f'Evaluation failed: {e}')
            raise EvaluationException(f'Evaluation failed: {e}')

    def validate_metric(self, 
                         metric: str) -> None:
        """
        Validate evaluation metric.

        Args:
        - metric (str): Evaluation metric.

        Raises:
        - InvalidMetricException: If metric is invalid.
        """
        if metric not in ['mse', 'mae']:
            raise InvalidMetricException(f'Invalid metric: {metric}')

def main():
    # Create evaluation configuration
    config = EvaluationConfig()

    # Create model
    model = torch.nn.Linear(10, 1)

    # Create device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create evaluator
    evaluator = Evaluator(config, model, device)

    # Create data
    data = pd.DataFrame(np.random.rand(100, 10))
    labels = pd.Series(np.random.rand(100))

    # Evaluate model
    metrics = evaluator.evaluate(data, labels)

    # Print metrics
    print(metrics)

if __name__ == '__main__':
    main()