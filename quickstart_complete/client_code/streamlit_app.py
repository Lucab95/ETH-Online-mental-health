import streamlit as st

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


import asyncio
async def upload_model():
    from dotenv import load_dotenv
    load_dotenv(f".env")
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
    return program_id, model_store_id, model_provider_party_id, permissions, model_secrets, payments_client, model_provider_client, model_provider_user_id, payments_wallet, cluster_id

def initialize_model():
    return asyncio.run(upload_model())

# Call the function to upload the model


# Now model_info contains all the returned values from upload_model()
if "model_info" not in st.session_state:
    # Run the model upload only once and store it in session_state
    st.write("Uploading model...")
    model_info = asyncio.run(upload_model())
    program_id, model_store_id, model_provider_party_id, permissions, model_secrets, payments_client, model_provider_client, model_provider_user_id, payments_wallet, cluster_id = model_info

    st.session_state.model_info = model_info
else:
    # Retrieve the model info from session state if it's already been uploaded
    model_info = st.session_state.model_info
    program_id, model_store_id, model_provider_party_id, permissions, model_secrets, payments_client, model_provider_client, model_provider_user_id, payments_wallet, cluster_id = model_info

import pandas as pd
import torch
import asyncio
from dotenv import load_dotenv
import nada_numpy as na
import nada_numpy.client as na_client
import py_nillion_client as nillion
from common.utils import compute, store_secrets
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from nillion_python_helpers import create_nillion_client, create_payments_config
from py_nillion_client import NodeKey, UserKey
import random
import string
import os
from nada_ai.client import TorchClient
import streamlit as st
def random_seed(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Your existing question sets
questions_dict = {
    'Q1A': "I found myself getting upset by quite trivial things.",
    'Q2A': "I was aware of dryness of my mouth.",
    'Q3A': "I couldn't seem to experience any positive feeling at all.",
    'Q4A': "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion).",
    'Q5A': "I just couldn't seem to get going.",
    'Q6A': "I tended to over-react to situations.",
    'Q7A': "I had a feeling of shakiness (e.g., legs going to give way).",
    'Q8A': "I found it difficult to relax.",
    'Q9A': "I found myself in situations that made me so anxious I was most relieved when they ended.",
    'Q10A': "I felt that I had nothing to look forward to.",
    'Q11A': "I found myself getting upset rather easily.",
    'Q12A': "I felt that I was using a lot of nervous energy.",
    'Q13A': "I felt sad and depressed.",
    'Q14A': "I found myself getting impatient when I was delayed in any way (e.g., elevators, traffic lights, being kept waiting).",
    'Q15A': "I had a feeling of faintness.",
    'Q16A': "I felt that I had lost interest in just about everything.",
    'Q17A': "I felt I wasn't worth much as a person.",
    'Q18A': "I felt that I was rather touchy.",
    'Q19A': "I perspired noticeably (e.g., hands sweaty) in the absence of high temperatures or physical exertion.",
    'Q20A': "I felt scared without any good reason.",
    'Q21A': "I felt that life wasn't worthwhile.",
    'Q22A': "I found it hard to wind down.",
    'Q23A': "I had difficulty in swallowing.",
    'Q24A': "I couldn't seem to get any enjoyment out of the things I did.",
    'Q25A': "I was aware of the action of my heart in the absence of physical exertion (e.g., sense of heart rate increase, heart missing a beat).",
    'Q26A': "I felt down-hearted and blue.",
    'Q27A': "I found that I was very irritable.",
    'Q28A': "I felt I was close to panic.",
    'Q29A': "I found it hard to calm down after something upset me.",
    'Q30A': "I feared that I would be 'thrown' by some trivial but unfamiliar task.",
    'Q31A': "I was unable to become enthusiastic about anything.",
    'Q32A': "I found it difficult to tolerate interruptions to what I was doing.",
    'Q33A': "I was in a state of nervous tension.",
    'Q34A': "I felt I was pretty worthless.",
    'Q35A': "I was intolerant of anything that kept me from getting on with what I was doing.",
    'Q36A': "I felt terrified.",
    'Q37A': "I could see nothing in the future to be hopeful about.",
    'Q38A': "I felt that life was meaningless.",
    'Q39A': "I found myself getting agitated.",
    'Q40A': "I was worried about situations in which I might panic and make a fool of myself.",
    'Q41A': "I experienced trembling (e.g., in the hands).",
    'Q42A': "I found it difficult to work up the initiative to do things."}

additional_questions_dict = {'TIPI1': "Extraverted, enthusiastic.",
    'TIPI2': "Critical, quarrelsome.",
    'TIPI3': "Dependable, self-disciplined.",
    'TIPI4': "Anxious, easily upset.",
    'TIPI5': "Open to new experiences, complex.",
    'TIPI6': "Reserved, quiet.",
    'TIPI7': "Sympathetic, warm.",
    'TIPI8': "Disorganized, careless.",
    'TIPI9': "Calm, emotionally stable.",
    'TIPI10': "Conventional, uncreative.",}

demographics_dict = {
    'religion': 'What is your religion? (1=Agnostic, 2=Atheist, 3=Buddhist, 4=Christian (Catholic), 5=Christian (Mormon), 6=Christian (Protestant), 7=Christian (Other), 8=Hindu, 9=Jewish, 10=Muslim, 11=Sikh, 12=Other)',
    'education': "How much education have you completed? (1=Less than high school, 2=High school, 3=University degree, 4=Graduate degree)",
    'urban': "What type of area did you live when you were a child? (1=Rural, 2=Suburban, 3=Urban)",
    'gender': "What is your gender? (1=Male, 2=Female, 3=Other)",
    'race': "What is your race? (1=Asian, 2=Arab, 3=Black, 4=Indigenous Australian, 5=Native American, 6=White, 7=Other)",
    'married': "What is your marital status? (1=Never married, 2=Currently married, 3=Previously married)",
    'age_group': "What is you age?",

    
    'familysize': 'Including you, how many children did your mother have?',
}

# Merge the dictionaries into a single one
questions_array = {**questions_dict, **additional_questions_dict, **demographics_dict}
question_list = list(questions_array.items())

# Initialize session state
if "responses" not in st.session_state:
    st.session_state.responses = {}

# Create a form
with st.form("questionnaire_form"):
    for current_question_key, current_question_text in question_list:
        if current_question_key in questions_dict:
            # Preselect the middle value (2) as the default
            options = [
                "1 - Did not apply to me at all",
                "2 - Applied to me to some degree, or some of the time",
                "3 - Applied to me to a considerable degree, or a good part of the time",
                "4 - Applied to me very much, or most of the time"
            ]
            default_value = 2
            st.markdown(f"<h3>{current_question_text}</h3>", unsafe_allow_html=True)  # Adjust <h3> to control size
            selected_option = st.radio(
                "", options, index=default_value - 1, key=current_question_key  # Leave the label empty
            )
            st.session_state.responses[current_question_key] = int(selected_option.split(" ")[0])
        elif current_question_key in additional_questions_dict:
            # Preselect the middle value (4) as the default
            options = [
                "1 - Disagree strongly",
                "2 - Disagree moderately",
                "3 - Disagree a little",
                "4 - Neither agree nor disagree",
                "5 - Agree a little",
                "6 - Agree moderately",
                "7 - Agree strongly"
            ]
            # default_value = 4
            # selected_option = st.radio(
            #     current_question_text, options, index=default_value - 1, key=current_question_key
            # )

            st.markdown(f"<h3>{current_question_text}</h3>", unsafe_allow_html=True)  # Adjust <h3> to control size
            selected_option = st.radio(
                "", options, index=default_value - 1, key=current_question_key  # Leave the label empty
            )
            st.session_state.responses[current_question_key] = int(selected_option.split(" ")[0])
        elif current_question_key == "religion":
            # Selectbox for religion
            options = [
                "1 - Agnostic", "2 - Atheist", "3 - Buddhist", "4 - Christian (Catholic)",
                "5 - Christian (Mormon)", "6 - Christian (Protestant)", "7 - Christian (Other)",
                "8 - Hindu", "9 - Jewish", "10 - Muslim", "11 - Sikh", "12 - Other"
            ]
            selected_option = st.selectbox(
                current_question_text, options, key=current_question_key
            )
            st.session_state.responses[current_question_key] = int(selected_option.split(" ")[0])
        elif current_question_key in demographics_dict:
            # Text input for demographics (familysize)
            response_value = st.text_input(current_question_text, key=current_question_key)
            if response_value.isdigit():
                st.session_state.responses[current_question_key] = int(response_value)
            else:
                st.session_state.responses[current_question_key] = 0  # Default to 0 if input is invalid
    
    # Submit button for the form
    submit_button = st.form_submit_button("Submit")

# Process and display the responses
if submit_button:
    st.write("Processing your input securely via Nillion...")
    # st.write(st.session_state.responses)  # Display the stored integer responses for verification
    input_df = pd.DataFrame([st.session_state.responses])

    # Prepare input for Nillion computation using st.session_state.responses
    input_df = pd.DataFrame([st.session_state.responses])

    # Reshape the input batch to match the expected input size for the model
    input_batch = input_df.values.reshape(1, -1)  # 1 row, N columns for N questions

    # Convert the input to the NADA format
    my_input_batch = na_client.array(input_batch, "my_input", na.SecretRational)

    async def run_nillion_computation():
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

        st.write(f"Processed result: {outputs}")
        class_mapping = {
        'Extremely Severe': 0,
        'Mild': 1,
        'Moderate': 2,
        'Normal': 3,
        'Severe': 4
    }

        # The mapping to the classes
        class_mapping = {
            'Extremely Severe': 0,
            'Mild': 1,
            'Moderate': 2,
            'Normal': 3,
            'Severe': 4
        }

        # Map the outputs to their corresponding classes
        output_mapping = {class_name: outputs[class_idx] for class_name, class_idx in class_mapping.items()}

        # Get the class with the highest chance
        highest_class = max(output_mapping, key=output_mapping.get)


        # Showing the highest class in a highlighted format
        if highest_class == "Normal":
            st.markdown(f"### Based on your responses, the model indicates **no depression symptoms**.")
        else:
            st.markdown(f"### Based on your responses, the model indicates **{highest_class} depressive symptoms**.")

                        

            # Run the async Nillion computation
    asyncio.run(run_nillion_computation())
