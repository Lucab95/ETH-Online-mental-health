import nada_numpy as na
from mental_health_nn import MentalHealthNN
from nada_dsl import *

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
    mental_health_model.load_state_from_network("mental_health_nn", provider, na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array((1, 60), user, "my_input", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model.forward(...)`
    result = mental_health_model(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(user, "my_output")