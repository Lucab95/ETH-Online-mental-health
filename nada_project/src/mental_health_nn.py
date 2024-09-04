import nada_numpy as na
from nada_ai import nn

class MentalHealthNN(nn.Module):
    """Mental health prediction model using fully connected layers for tabular data"""

    def __init__(self, input_size: int, num_classes: int) -> None:
        """Model is composed of fully connected layers for tabular data"""
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, 16)          # Second fully connected layer
        self.fc3 = nn.Linear(16, 8)          # Third fully connected layer
        self.fc4 = nn.Linear(8, num_classes) # Output layer
        self.relu = nn.ReLU()

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Forward pass logic for fully connected layers"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
