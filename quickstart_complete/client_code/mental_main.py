import torch
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
class MentalHealthDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.data = dataframe.drop(['target', 'total_count'], axis=1)
        self.targets = dataframe['target']
        
        self.label_encoder = LabelEncoder()
        self.targets = self.label_encoder.fit_transform(self.targets)
        
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.targets[index], dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label


import torch.nn as nn
import nada_numpy as na  # Assuming na.NadaArray usage is required by the NADA environment

class MentalHealthNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MentalHealthNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # First fully connected layer
        self.fc2 = nn.Linear(32, 16)          # Second fully connected layer
        self.fc3 = nn.Linear(16, 8)          # Third fully connected layer
        self.fc4 = nn.Linear(8, num_classes) # Output layer
        self.relu = nn.ReLU()

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Load your dataset
depression = pd.read_csv('depression_dataset.csv')

# Initialize dataset
dataset = MentalHealthDataset(depression)

# Split into train and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize model, loss function, and optimizer
input_size = dataset.data.shape[1]
num_classes = len(dataset.label_encoder.classes_)
model = MentalHealthNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

train_accuracies = []
val_accuracies = []
train_losses = []

from dotenv import load_dotenv
load_dotenv(f".env")
async def main() -> None:

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    import random

    import asyncio

    import nada_numpy as na
    import nada_numpy.client as na_client
    import numpy as np
    import py_nillion_client as nillion
    import torch
    from common.utils import compute, store_program, store_secrets
    from cosmpy.aerial.client import LedgerClient
    from cosmpy.aerial.wallet import LocalWallet
    from cosmpy.crypto.keypairs import PrivateKey
    from dotenv import load_dotenv
    from nillion_python_helpers import (create_nillion_client,
                                        create_payments_config)
    from py_nillion_client import NodeKey, UserKey

    from nada_ai.client import TorchClient

    # seed = str(random.randint(1, 1000))
    # seed = "mwery_d"
    # model_provider_userkey = UserKey.from_seed((seed))
    # model_provider_nodekey = NodeKey.from_seed((seed))
    import random
    import string

    def random_seed(length=8):
        # Generate a random string of letters (both uppercase and lowercase) and digits
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    seed = random_seed()
    model_provider_userkey = UserKey.from_seed((seed))
    model_provider_nodekey = NodeKey.from_seed((seed))

    print("Randomized seed:", seed)

    model_provider_client = create_nillion_client(model_provider_userkey, model_provider_nodekey)
    model_provider_party_id = model_provider_client.party_id
    model_provider_user_id = model_provider_client.user_id

    party_names = ["Provider", "User"]
    program_name = "mental_main"
    program_mir_path = f"../nada_quickstart_programs/target/{program_name}.nada.bin"

    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    program_id = await store_program(
        model_provider_client,
        payments_wallet,
        payments_client,
        model_provider_user_id,
        cluster_id,
        program_name,
        program_mir_path,)

    model_client = TorchClient(model)

    model_secrets = nillion.NadaValues(
        model_client.export_state_as_secrets("mental_main", na.SecretRational)
    )

    print("Model secrets being stored:", model_secrets)

    permissions = nillion.Permissions.default_for_user(model_provider_client.user_id)
    permissions.add_compute_permissions({model_provider_client.user_id: {program_id}})

    model_store_id = await store_secrets(
        model_provider_client,
        payments_wallet,
        payments_client,
        cluster_id,
        model_secrets,
        1,
        permissions,
    )
    print("saving model")
# # This information is needed by the model user
#     with open("src/data/tmp.json", "w") as provider_variables_file:
#         provider_variables = {
#             "program_id": program_id,
#             "model_store_id": model_store_id,
#             "model_provider_party_id": model_provider_party_id,
#         }
    print( {
            "program_id": program_id,
            "model_store_id": model_store_id,
            "model_provider_party_id": model_provider_party_id,
        })
    import json
    # json.dump(provider_variables, provider_variables_file)  # This should be inside the with block

    print('model fully stored')

    csv_file_path = 'depression_dataset.csv'  # Path to your dataset
    data = pd.read_csv(csv_file_path)

    # Assuming 'target' and 'total_count' are columns you don't need
    features = data.drop(['target', 'total_count'], axis=1)  
    targets = data['target']  # If you need labels, extract them separately

    # Prepare batches of inputs, here for demonstration we'll just split the first N rows
    batch_size = 32
    input_batch = features.values[:batch_size]  # Take the first 32 rows as a batch

    # Reshape the input batch to match the expected input size for the model
    input_batch_reshaped = input_batch.reshape(batch_size, -1)  # Shape (32, 60) for example

    # Convert the batch to the custom distributed framework format
    my_input_batch = na_client.array(input_batch_reshaped, "my_input", na.SecretRational)

    input_secrets_batch = nillion.NadaValues(my_input_batch)

    data_store_id = await store_secrets(
        model_provider_client,
        payments_wallet,
        payments_client,
        cluster_id,
        input_secrets_batch,
        1,
        permissions,
    )


    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party("Provider", model_provider_client.party_id)
    compute_bindings.add_input_party("User", model_provider_client.party_id)  # Ensure User's party_id is correct

    compute_bindings.add_output_party("User", model_provider_client.party_id)

    print("Provider", model_provider_client.party_id)
    print("User", model_provider_client.party_id)
    print("compute_bindings", compute_bindings)
    

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {model_store_id}, {data_store_id}")


    computation_time_secrets = nillion.NadaValues({})
    result = await compute(
        model_provider_client,
        payments_wallet,
        payments_client,
        program_id,
        cluster_id,
        compute_bindings,
        [model_store_id, data_store_id],
        nillion.NadaValues({}),
        verbose=True,
    )
    
    print(result)
    outputs = [
        na_client.float_from_rational(result[1])
        for result in sorted(
            result.items(),
            key=lambda x: int(x[0].replace("my_output", "").replace("_", "")),
        )
    ]

    print(f"🖥️  The processed result is {outputs} @ {na.get_log_scale()}-bit precision")
import asyncio



if __name__ == "__main__":
    asyncio.run(main())