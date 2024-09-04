import pandas as pd
import nada_numpy as na
from src.mental_health_nn import MentalHealthNN
from common.utils import *
from nada_dsl import *
from sklearn.preprocessing import MinMaxScaler
import json
import asyncio
import numpy as np
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey
import py_nillion_client as nillion
import os
import random
from cosmpy.crypto.keypairs import PrivateKey

async def store_features(model_user_client, payments_wallet, payments_client, cluster_id, input_data, secret_name, nada_type, ttl_days, permissions):
    features_store_id = await store_secret_array(
        model_user_client,
        payments_wallet,
        payments_client,
        cluster_id,
        input_data,
        secret_name,
        nada_type,
        ttl_days,
        permissions,
    )
    return features_store_id

async def compute_results(model_user_client, payments_wallet, payments_client, program_id, cluster_id, compute_bindings, model_store_id, features_store_id):
    result = await compute(
        model_user_client,
        payments_wallet,
        payments_client,
        program_id,
        cluster_id,
        compute_bindings,
        [model_store_id, features_store_id],
        nillion.NadaValues({}),
        verbose=True,
    )
    return result

def main():
    # Step 1: We use Nada NumPy wrapper to create "User" and "Provider"
    user = Party("User")
    provider = Party("Provider")

    # Step 2: Load the CSV file containing the poll data
    csv_file_path = '/home/brglt/Desktop/nillion/ETH-Online-mental-health/nada_project/depression_dataset copy.csv'  # Update with your actual file path
    
    data = pd.read_csv(csv_file_path)
    data = data[:1]
    print(data)
    # Step 3: Process the data to prepare it for inference (assuming we are dropping the 'target' column)
    features = data.drop(['target', 'total_count'], axis=1)  # Adjust this if you have a target column

    # Step 4: Preprocess using MinMaxScaler to normalize the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Step 5: Select a single sample from the CSV (or process the entire batch) and convert to (1, 60)
    single_input = np.array(scaled_features[0].reshape(1, -1))  # Selecting the first sample for demonstration

    # Step 6: Instantiate the MentalHealthNN model object (set the input size and num classes)
    input_size = 60  # As we need (1, 60) input size
    num_classes = 5  # Adjust according to your model's requirements
    mental_health_model = MentalHealthNN(input_size, num_classes)

    # Step 7: Load model weights from Nillion network
    mental_health_model.load_state_from_network("mental_health_nn", provider, na.SecretRational)

    # Step 8: Store the input data in the Nillion network
    model_user_userkey = UserKey.from_seed("abc")
    model_user_nodekey = NodeKey.from_seed(str(random.randint(0, 1000)))

    print("model_user_userkey:", model_user_userkey)
    print("model_user_nodekey:", model_user_nodekey)
    # Create Nillion client
    model_user_client = create_nillion_client(model_user_userkey, model_user_nodekey)

    # Define permissions and other parameters
    permissions = nillion.Permissions.default_for_user(model_user_client.user_id)

    # Payments configuration
    payments_config = create_payments_config(os.getenv("NILLION_NILCHAIN_CHAIN_ID"), os.getenv("NILLION_NILCHAIN_GRPC"))
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))), prefix="nillion")

    # Load program variables
    with open("src/data/tmp.json", "r") as provider_variables_file:
        provider_variables = json.load(provider_variables_file)

    program_id = provider_variables["program_id"]
    model_store_id = provider_variables["model_store_id"]
    cluster_id = os.getenv("NILLION_CLUSTER_ID")

    # Store the features in the Nillion network (async call)
    features_store_id = asyncio.run(store_features(model_user_client, payments_wallet, payments_client, cluster_id, single_input, "my_input", na.SecretRational, 1, permissions))

    # Step 9: Set up the compute bindings and run inference
    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party("Provider", provider.user_id)
    compute_bindings.add_input_party("User", model_user_client.party_id)
    compute_bindings.add_output_party("User", model_user_client.party_id)

    # Run the computation
    result = asyncio.run(compute_results(model_user_client, payments_wallet, payments_client, program_id, cluster_id, compute_bindings, model_store_id, features_store_id))

    # Step 10: Return the result of the computation
    first_key = next(iter(result))
    output_value = result[first_key]
    return output_value.output(user, "my_output")

if __name__ == "__main__":
    main()