import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

################################################
## ---- Features ----
id_col = 'id'
date_col = 'start_date'
numerical_cols = ['stringency_index',
                  # 'year',
                  # 'month_sin',
                  # 'month_cos'
                  ]
categorical_cols = ['phase',
                    'allocation',
                    'intervention_model',
                    'primary_purpose',
                    'dmc_oversight',
                    'fda_drug',
                    'fda_device',
                    'unapproved_device'
                    ]
categorical_embedding_dims = [(7, 4), (4, 2), (6, 3), (10, 5), (4, 2), (4, 2), (4, 2), (3, 2)]
target_term_col = 'terminated'
target_score_col = 'ae_score'


## ---- JSON Flattening ----
def flatten_json(nested_json, flat_json=None, prefix=''):
    ''' Flatten JSON to appropriate level, accounting for preservation keys '''

    # Preserve in original form, do not flatten
    preserve_list = [
    'protocolSection_conditionsModule_conditions',
    'protocolSection_conditionsModule_keywords',
    'protocolSection_armsInterventionsModule_armGroups',
    'protocolSection_armsInterventionsModule_interventions',
    'protocolSection_outcomesModule_primaryOutcomes',
    'protocolSection_outcomesModule_secondaryOutcomes',
    'protocolSection_outcomesModule_otherOutcomes',
    'resultsSection_adverseEventsModule_eventGroups',
    'resultsSection_adverseEventsModule_seriousEvents',
    'resultsSection_adverseEventsModule_otherEvents'
    ]

    if flat_json is None:
        flat_json = {}

    if isinstance(nested_json, dict):
        for k, v in nested_json.items():
            new_prefix = f"{prefix}{k}_" if prefix else f"{k}_"
            if new_prefix[:-1] in preserve_list and (isinstance(v, dict) or isinstance(v, list)):
                flat_json[new_prefix[:-1]] = v
            else:
                flatten_json(v, flat_json, new_prefix)
    elif isinstance(nested_json, list):
        for i, item in enumerate(nested_json):
            new_prefix = f"{prefix}{i}_"
            flatten_json(item, flat_json, new_prefix)
    else:
        flat_json[prefix[:-1]] = nested_json

    return flat_json


################################################
## ---- Text Accumulation ----
def printList(item, is_first_list):
    s = ''
    if isinstance(item, dict):
        for k, v in item.items():
            s += str(k.capitalize()) + ': ' + printList(v, False) + '\n'
    if isinstance(item, list):
        for i in range(len(item)):
            if is_first_list:
                s += str(i + 1) + ". " + printList(item[i], False)
            elif i == len(item) - 1:
                s += printList(item[i], False)
            else:
                s += printList(item[i], False) + ', '
    if isinstance(item, str):
        s += item
    return s

def clean_criteria(text):
    text = text.replace('\n\n', '\n')

    exclusion_index = text.find("Exclusion Criteria")
    if exclusion_index == -1:
        return text  # "Exclusion Criteria" not found

    last_newline_index = text.rfind("\n", 0, exclusion_index)
    if last_newline_index == -1:
        return text  # No newline found before "Exclusion Criteria"

    return text[:last_newline_index] + "\n\n" + text[last_newline_index + 1:]

def accumulate_text_intro(trial):
    s = ''
    for key, value in trial.items():
        if key in [
            # 'protocolSection_identificationModule_officialTitle',
            'protocolSection_descriptionModule_briefSummary',
            'protocolSection_descriptionModule_detailedDescription'
        ]:
            s += value + '\n'

        if key == 'protocolSection_conditionsModule_conditions':
            s += '\nConditions: ' + printList(value, False) + '\n'

        if key == 'protocolSection_armsInterventionsModule_armGroups':
            s += '\nTrial Arms: ' + '\n' + printList(value, True) + '\n'
        if key == 'protocolSection_armsInterventionsModule_interventions':
            s += 'Trial Interventions: ' + '\n' + printList(value, True) + '\n'
    return s

def accumulate_text_outcomes(trial):
    s = ''
    for key, value in trial.items():
        if key == 'protocolSection_outcomesModule_primaryOutcomes':
            s += 'Primary Outcomes: ' + '\n' + printList(value, True) + '\n'
        if key == 'protocolSection_outcomesModule_secondaryOutcomes':
            s += 'Secondary Outcomes: ' + '\n' + printList(value, True) + '\n'
    return s

def accumulate_text_criteria(trial):
    s = ''
    for key, value in trial.items():
        if key == 'protocolSection_eligibilityModule_eligibilityCriteria':
            s += clean_criteria(value) + '\n\n'   # Remove excess spacing
        if key == 'protocolSection_eligibilityModule_sex':
            s += 'Sex: ' + value + '\n'
        if key == 'protocolSection_eligibilityModule_minimumAge':
            s += 'Minimum Age: ' + value + '\n'
        if key == 'protocolSection_eligibilityModule_maximumAge':
            s += 'Maxmium Age: ' + value + '\n'
    return s


################################################
## --- Create Dataframe ---
def create_df(trials):
    df = pd.DataFrame(trials)

    # Set date type
    # Convert to datetime with errors='coerce'
    df[date_col] = pd.to_datetime(df['protocolSection_statusModule_startDateStruct_date'], errors='coerce')
    # Find rows where start_date is NaT (YYYY-MM dates)
    nat_rows = df[date_col].isna()
    # Create a copy of the original string column
    temp_date_strings = df['protocolSection_statusModule_startDateStruct_date'].copy()
    # Append '-01' to NaT rows
    temp_date_strings[nat_rows] = temp_date_strings[nat_rows] + '-01'
    # Convert the modified strings to datetime
    df[date_col] = df[date_col].fillna(pd.to_datetime(temp_date_strings, errors='coerce'))

    # Keep Necessary Columns
    keep_col = [
    'protocolSection_identificationModule_nctId',
    'protocolSection_statusModule_overallStatus',
    'protocolSection_oversightModule_oversightHasDmc',
    'protocolSection_oversightModule_isFdaRegulatedDrug',
    'protocolSection_oversightModule_isFdaRegulatedDevice',
    'protocolSection_oversightModule_isUnapprovedDevice',
    'protocolSection_designModule_phases_0',
    'protocolSection_designModule_phases_1',
    'protocolSection_designModule_designInfo_allocation',
    'protocolSection_designModule_designInfo_interventionModel',
    'protocolSection_designModule_designInfo_primaryPurpose',
    'hasResults',
    'acc_text',
    'text_intro',
    'text_outcomes',
    'text_criteria',
    'sae_count',
    'other_ae_count',
    'sae_events',
    'other_ae_events',
    'ae_risk_score',
    'protocolSection_statusModule_whyStopped',
    'resultsSection_adverseEventsModule_eventGroups',
    'resultsSection_adverseEventsModule_seriousEvents',
    'resultsSection_adverseEventsModule_otherEvents',
    ]

    # Keep columns in list, ignore if not in list
    keep_col = [col for col in keep_col if col in df.columns]
    df = df[keep_col]

    # Rename columns
    df.rename(columns={
        'protocolSection_identificationModule_nctId': id_col,
        'protocolSection_statusModule_overallStatus': 'status',
        'protocolSection_oversightModule_oversightHasDmc': 'dmc_oversight',
        'protocolSection_oversightModule_isFdaRegulatedDrug': 'fda_drug',
        'protocolSection_oversightModule_isFdaRegulatedDevice': 'fda_device',
        'protocolSection_oversightModule_isUnapprovedDevice': 'unapproved_device',
        'protocolSection_designModule_phases_0': 'phase',
        'protocolSection_designModule_designInfo_allocation': 'allocation',
        'protocolSection_designModule_designInfo_interventionModel': 'intervention_model',
        'protocolSection_designModule_designInfo_primaryPurpose': 'primary_purpose',
        'protocolSection_designModule_enrollmentInfo_count' : 'enroll_count',
    }, errors='ignore', inplace=True)


    # --- Add Required Features ---

    # Add assumed stringency index. Future case may need to have the stringency index updated to reflect current political conditions
    df['stringency_index'] = 0

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = np.nan
            df[col] = df[col].astype('object')  # Ensure object dtype for consistency

    return df


def fill_nan_categorical(df, categorical_cols, fill_value="null"):
    for col in categorical_cols:
        # Replace actual np.nan
        df[col] = df[col].fillna(fill_value)
        # Replace literal 'NA' string
        df[col] = df[col].replace('NA', fill_value) # Handle cases of literal 'NA', which can cause inconsistency when csv is saved and loaded



################################################
## ---- Tokenize ----

# Check if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bb_model = AutoModel.from_pretrained(model_name)
# Move the model to GPU if available, otherwise CPU
bb_model = bb_model.to(device)

def tokenize_text_sections(df, tokenizer=None, max_length=512, batch_size=128):

    all_intro_input_ids = []
    all_intro_attention_masks = []
    all_outcomes_input_ids = []
    all_outcomes_attention_masks = []
    all_criteria_input_ids = []
    all_criteria_attention_masks = []

    # Create batch
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size].copy() # Batch dataframe
        text_intros_batch = batch_df['text_intro'].tolist() # Batch of text rows
        text_outcomes_batch = batch_df['text_outcomes'].tolist()
        text_criteria_batch = batch_df['text_criteria'].tolist()

        # Batch tokenize the texts.
        # Tokenize intros
        encoded_intro_batch = tokenizer(text_intros_batch,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,  # Let tokenizer add CLS and SEP
            return_attention_mask=True,
        )
        all_intro_input_ids.extend(encoded_intro_batch['input_ids'])
        all_intro_attention_masks.extend(encoded_intro_batch['attention_mask'])

        # Tokenize outcomes
        encoded_outcomes_batch = tokenizer(text_outcomes_batch,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,  # Let tokenizer add CLS and SEP
            return_attention_mask=True,
        )
        all_outcomes_input_ids.extend(encoded_outcomes_batch['input_ids'])
        all_outcomes_attention_masks.extend(encoded_outcomes_batch['attention_mask'])

        # Tokenize criteria
        encoded_criteria_batch = tokenizer(text_criteria_batch,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,  # Let tokenizer add CLS and SEP
            return_attention_mask=True,
        )
        all_criteria_input_ids.extend(encoded_criteria_batch['input_ids'])
        all_criteria_attention_masks.extend(encoded_criteria_batch['attention_mask'])


    df['intro_input_ids'] = all_intro_input_ids
    df['intro_attention_mask'] = all_intro_attention_masks
    df['outcomes_input_ids'] = all_outcomes_input_ids
    df['outcomes_attention_mask'] = all_outcomes_attention_masks
    df['criteria_input_ids'] = all_criteria_input_ids
    df['criteria_attention_mask'] = all_criteria_attention_masks

    return df


################################################
## ---- Pipeline Transform ---

# Load numerical scalers
numerical_scaler_pickle_filename ='numerical_scalers.pkl'
try:
    with open(numerical_scaler_pickle_filename, 'rb') as f:
        numerical_scalers = pickle.load(f)
    print(f"Numerical scalers loaded from (pickle): {numerical_scaler_pickle_filename}")
    print(f"Loaded numerical scalers (pickle) type: {type(numerical_scalers)}")
# Use numerical_scalers to transform new data
except FileNotFoundError:
    print(f"Error: File not found: {numerical_scaler_pickle_filename}")
except Exception as e:
    print(f"Error loading numerical scalers (pickle): {e}")

# Load categorical mappings
categorical_mapping_pickle_filename = 'categorical_mappings.pkl'
try:
    with open(categorical_mapping_pickle_filename, 'rb') as f:
        categorical_mappings = pickle.load(f)
    print(f"Categorical mappings loaded from (pickle): {categorical_mapping_pickle_filename}")
    print(f"Loaded categorical mappings (pickle) type: {type(categorical_mappings)}")
    # Use categorical_mappings to map new categories
except FileNotFoundError:
    print(f"Error: File not found: {categorical_mapping_pickle_filename}")
except Exception as e:
    print(f"Error loading categorical mappings (pickle): {e}")


# Pipeline transform data
def pipeline_transform(df, numerical_scalers, categorical_mappings):
    ''' Iterate through the scalers and mappings to modify relevant features
    A KeyError would indicate column missing in the dataframe '''

    # Transform numerical columns (iterate through scaler keys)
    for col in numerical_scalers:
        df[col] = numerical_scalers[col].transform(df[[col]])  # Transform single column

    # Transform categorical columns (iterate through mapping keys)
    for col in categorical_mappings:
        mapping = categorical_mappings[col]
        df[col] = df[col].apply(lambda x: mapping.get(x, 0))  # 0 to handle unseen values

    return df