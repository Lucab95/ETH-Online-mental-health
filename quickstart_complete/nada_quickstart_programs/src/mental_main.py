import nada_numpy as na

from nada_dsl import *
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

def nada_main():
    # Step 1: We use Nada NumPy wrapper to create "User" and "Provider"
    user = Party("User")
    provider = Party("Provider")

    # Step 2: Instantiate model object
    input_size = 60
    num_classes = 5
    mental_health_model = MentalHealthNN(input_size, num_classes)

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    mental_health_model.load_state_from_network("mental_main", provider, na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array((1, 60), user, "my_input", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model.forward(...)`
    result = mental_health_model(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(user, "my_output")