import json
from flask import Flask, request, jsonify

from inference_preprocess import *
from trial_models import TrialDataset, Terminate_Model, AE_Score_Model


try:
    with open('row0_clinicaltrials_response.json', 'r') as f:
        data = json.load(f)

    if "studies" in data:
        trials = data["studies"]
        # Count only trials that actually contain 'protocolSection'
        valid_trials = [t for t in trials if "protocolSection" in t]
        print("Total Trials in 'studies':", len(trials))
        print("Trials with 'protocolSection':", len(valid_trials))

except FileNotFoundError:
    print(f"Error: File not found.")
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format.")
except Exception as e:
    print(f"An unexpected error occured: {e}")


################################################
## ---- Preprocess Data ----

# Flatten trials jsons
flattened_trials = []
for trial in trials:
    flattened_data = flatten_json(trial)
    flattened_trials.append(flattened_data)

# Accumulate and section text
for trial in flattened_trials:
    trial['text_intro'] = accumulate_text_intro(trial)
    trial['text_outcomes'] = accumulate_text_outcomes(trial)
    trial['text_criteria'] = accumulate_text_criteria(trial)

# Create dataframe
df = create_df(flattened_trials)
# Explicitly define null
fill_nan_categorical(df, categorical_cols)

# Tokenize Text
df = tokenize_text_sections(df)

# Transform
df = pipeline_transform(df,
                        numerical_scalers=numerical_scalers,
                        categorical_mappings=categorical_mappings
                        )


################################################
## ---- Construct Torch Dataset ----

# Construct torch dataset
trial_data = TrialDataset(df,
                        id_col=id_col,
                        date_col=date_col,
                        categorical_cols=categorical_cols,
                        numerical_cols=numerical_cols,
                        intro_ids_col='intro_input_ids', intro_mask_col='intro_attention_mask',
                        outcomes_ids_col='outcomes_input_ids', outcomes_mask_col='outcomes_attention_mask',
                        criteria_ids_col='criteria_input_ids', criteria_mask_col='criteria_attention_mask',
                        )

# Construct torch dataloader
batch_size = max(len(df), 10) # Not expecting anything larger than 10 realistically
input_dataloader = DataLoader(trial_data, batch_size, shuffle=False)


################################################
## ---- Construct Model and Import Weights ----

# Load Terminate Model
terminate_model = Terminate_Model(
                        num_categorical_features=len(categorical_cols),
                        categorical_embedding_dims=categorical_embedding_dims,
                        num_numerical_features=len(numerical_cols),
                        embed_model=bb_model)
state_dict = torch.load('model_terminate_state_dict.pth', map_location= 'cpu')
terminate_model.load_state_dict(state_dict)
terminate_model.to(device)
terminate_model.eval()

# Load AE Score Model
ae_score_model = AE_Score_Model(
                        num_categorical_features=len(categorical_cols),
                        categorical_embedding_dims=categorical_embedding_dims,
                        num_numerical_features=len(numerical_cols),
                        embed_model=bb_model)

state_dict = torch.load('model_ae_score_state_dict.pth', map_location= 'cpu')
ae_score_model.load_state_dict(state_dict)
ae_score_model.to(device)
ae_score_model.eval()


## ---- Predict ----
# Predict with Terminate Model
prob_terminations = []
with torch.no_grad(): # Disable gradient calculations during validation
    for batch_idx, batch in enumerate(input_dataloader):
        intro_input_ids_batch = batch['intro_input_ids'].to(device)
        intro_attention_mask_batch = batch['intro_attention_mask'].to(device)
        outcomes_input_ids_batch = batch['outcomes_input_ids'].to(device)
        outcomes_attention_mask_batch = batch['outcomes_attention_mask'].to(device)
        criteria_input_ids_batch = batch['criteria_input_ids'].to(device)
        criteria_attention_mask_batch = batch['criteria_attention_mask'].to(device)
        categorical_batch = batch['categorical_inputs'].to(device)
        numerical_batch = batch['numerical_inputs'].to(device)

    # Forward pass of validation data
        outputs = terminate_model(categorical_batch, numerical_batch,
                intro_input_ids_batch, intro_attention_mask_batch,
                outcomes_input_ids_batch, outcomes_attention_mask_batch,
                criteria_input_ids_batch, criteria_attention_mask_batch)

        # Take sigmoid output as probability of termination class 1
        prob_terminate = torch.sigmoid(outputs[:, 1])
        prob_terminations.extend(prob_terminate.cpu().numpy())

# Convert numpy.float32 to regular Python float before serializing
prob_terminations = [float(x) for x in prob_terminations]
# Prepare the result to be sent as JSON
prob_terminations_results = {"predictions": prob_terminations}


# Predict with AE Score Model
ae_risks = []
with torch.no_grad(): # Disable gradient calculations during validation
    for batch_idx, batch in enumerate(input_dataloader):
        intro_input_ids_batch = batch['intro_input_ids'].to(device)
        intro_attention_mask_batch = batch['intro_attention_mask'].to(device)
        outcomes_input_ids_batch = batch['outcomes_input_ids'].to(device)
        outcomes_attention_mask_batch = batch['outcomes_attention_mask'].to(device)
        criteria_input_ids_batch = batch['criteria_input_ids'].to(device)
        criteria_attention_mask_batch = batch['criteria_attention_mask'].to(device)
        categorical_batch = batch['categorical_inputs'].to(device)
        numerical_batch = batch['numerical_inputs'].to(device)

    # Forward pass of validation data
        outputs = ae_score_model(categorical_batch, numerical_batch,
                intro_input_ids_batch, intro_attention_mask_batch,
                outcomes_input_ids_batch, outcomes_attention_mask_batch,
                criteria_input_ids_batch, criteria_attention_mask_batch)

        # Clamp to [0, 1.33] and normalize to [0, 1]
        ae_risk = (torch.clamp(outputs, min=0.0, max=1.33)) / 1.33
        ae_risks.extend(ae_risk.cpu().numpy().tolist())

# Convert numpy.float32 to regular Python float before serializing
ae_risks = [float(x) for x in ae_risks]
# Prepare the result to be sent as JSON
ae_risks_results = {"predictions": ae_risks}

