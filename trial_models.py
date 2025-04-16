import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

################################################
# ---- Define PyTorch Dataset class ----
class TrialDataset(Dataset):
    def __init__(self, dataframe, id_col, date_col, categorical_cols, numerical_cols,
                 intro_ids_col, intro_mask_col,
                 outcomes_ids_col, outcomes_mask_col,
                 criteria_ids_col, criteria_mask_col,
                 target_term_col=None, target_score_col=None):
        self.id_col = id_col
        self.date_col = date_col
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.intro_ids_col = intro_ids_col
        self.intro_mask_col = intro_mask_col
        self.outcomes_ids_col = outcomes_ids_col
        self.outcomes_mask_col = outcomes_mask_col
        self.criteria_ids_col = criteria_ids_col
        self.criteria_mask_col = criteria_mask_col
        self.target_term_col = target_term_col
        self.target_score_col = target_score_col
        #
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # For dataloader
        item = self.data.iloc[idx]
        trial_id = item[self.id_col]
        intro_input_ids = torch.tensor(item[self.intro_ids_col], dtype=torch.long)
        intro_attention_mask = torch.tensor(item[self.intro_mask_col], dtype=torch.long)
        outcomes_input_ids = torch.tensor(item[self.outcomes_ids_col], dtype=torch.long)
        outcomes_attention_mask = torch.tensor(item[self.outcomes_mask_col], dtype=torch.long)
        criteria_input_ids = torch.tensor(item[self.criteria_ids_col], dtype=torch.long)
        criteria_attention_mask = torch.tensor(item[self.criteria_mask_col], dtype=torch.long)
        categorical_inputs = torch.tensor([item[col] for col in self.categorical_cols], dtype=torch.long)
        numerical_inputs = torch.tensor([item[col] for col in self.numerical_cols], dtype=torch.float)

        dloader_dict =  {
            'id': trial_id,
            'intro_input_ids': intro_input_ids,
            'intro_attention_mask': intro_attention_mask,
            'outcomes_input_ids': outcomes_input_ids,
            'outcomes_attention_mask': outcomes_attention_mask,
            'criteria_input_ids': criteria_input_ids,
            'criteria_attention_mask': criteria_attention_mask,
            'categorical_inputs': categorical_inputs,
            'numerical_inputs': numerical_inputs
        }
        # Target is optional to account for new-world data
        if self.target_term_col is not None:   
            target = torch.tensor(item[self.target_term_col], dtype=torch.long)
            dloader_dict['targets_term'] = target
        if self.target_score_col is not None:   
            target = torch.tensor(item[self.target_score_col], dtype=torch.float)
            dloader_dict['targets_score'] = target

        return dloader_dict



###############################################
## ---- Terminate Model ----

class Terminate_Model(nn.Module):
    def __init__(self,
                 num_categorical_features=None,
                 categorical_embedding_dims=[],
                 num_numerical_features=None,
                 embed_model=None):

        super(Terminate_Model, self).__init__()
        self.biobert = embed_model

        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for num_embeddings, embedding_dim in categorical_embedding_dims
        ])
        self.num_categorical_features = num_categorical_features

        self.numerical_bn = nn.BatchNorm1d(num_numerical_features)  # Creates and assigns layer (a method) designed for given number of numerical features
        self.num_numerical_features = num_numerical_features

        # Combine features
        combined_input_dim = 3 * self.biobert.config.hidden_size  # 3 Outputs of BioBERT
        if num_categorical_features is not None:
            combined_input_dim += sum([dim for _, dim in categorical_embedding_dims])
        if num_numerical_features is not None:
            combined_input_dim += num_numerical_features

        # Weighting layers
        # Modality weights: 3 CLS embeddings
        self.text_modality_weights = nn.Parameter(torch.ones(3))  # shape (3,)
        # Per-feature categorical weights
        self.categorical_feature_weights = nn.Parameter(torch.ones(num_categorical_features))  # shape (num_categorical_features,)
        # Single numerical feature
        self.numerical_feature_weights = nn.Parameter(torch.ones(num_numerical_features)) # shape (num_numerical_features,)

        # Layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(combined_input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512) # Second hidden layer
        self.bn2 = nn.BatchNorm1d(512) # BatchNorm for the second hidden layer
        self.linear3 = nn.Linear(512, 256) # Second hidden layer
        self.bn3 = nn.BatchNorm1d(256) # BatchNorm for the second hidden layer
        self.finallinear = nn.Linear(256, 2) # Output layer


    def forward(self, categorical_inputs, numerical_inputs,
                intro_input_ids, intro_attention_mask,
                outcomes_input_ids, outcomes_attention_mask,
                criteria_input_ids, criteria_attention_mask):

        # Embed each text input using BioBERT
        intro_outputs = self.biobert(intro_input_ids, attention_mask=intro_attention_mask)
        intro_embedding = intro_outputs.pooler_output

        outcomes_outputs = self.biobert(outcomes_input_ids, attention_mask=outcomes_attention_mask)
        outcomes_embedding = outcomes_outputs.pooler_output

        criteria_outputs = self.biobert(criteria_input_ids, attention_mask=criteria_attention_mask)
        criteria_embedding = criteria_outputs.pooler_output

        ## --- Concatenate ---
        # Concatenate the embeddings from the three text inputs
        # Apply modality weights (softmax optional for normalized weight distribution)
        modality_weights = F.softmax(self.text_modality_weights, dim=0)
        text_embeddings = [
            intro_embedding * modality_weights[0],
            outcomes_embedding * modality_weights[1],
            criteria_embedding * modality_weights[2]
        ]
        text_concat = torch.cat(text_embeddings, dim=1)  # (batch, 2304)

        # Categorical: embed and apply per-feature weights
        if self.num_categorical_features is not None:
          categorical_embeds = [emb(categorical_inputs[:, i]) for i, emb in enumerate(self.categorical_embeddings)]
          categorical_embeds = [embed * self.categorical_feature_weights[i] for i, embed in enumerate(categorical_embeds)]
          cat_concat = torch.cat(categorical_embeds, dim=1)  # shape (batch, sum(emb_dims))

        # Numerical
        if self.num_numerical_features is not None:
            numerical_inputs = self.numerical_bn(numerical_inputs)  # Input numerical inputs to batch normalization layer
            # numerical_feature_weights: (num_features,) → auto-broadcast to (batch, num_features)
            weighted_numerical = numerical_inputs * self.numerical_feature_weights

        # Final concat
        combined_features = torch.cat([text_concat, cat_concat, weighted_numerical], dim=1)


        # Forward
        x = self.dropout1(combined_features)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout2(x)
        logits = self.finallinear(x)
        return logits


###############################################
## ---- AE Score Model ----
class AE_Score_Model(nn.Module):
    def __init__(self,
                 num_categorical_features=None,
                 categorical_embedding_dims=[],
                 num_numerical_features=None,
                 embed_model=None):

        super(AE_Score_Model, self).__init__()
        self.biobert = embed_model

        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for num_embeddings, embedding_dim in categorical_embedding_dims
        ])
        self.num_categorical_features = num_categorical_features

        self.numerical_bn = nn.BatchNorm1d(num_numerical_features)  # Creates and assigns layer (a method) designed for given number of numerical features
        self.num_numerical_features = num_numerical_features

        # Combine features
        combined_input_dim = 3 * self.biobert.config.hidden_size  # 3 Outputs of BioBERT
        if num_categorical_features is not None:
            combined_input_dim += sum([dim for _, dim in categorical_embedding_dims])
        if num_numerical_features is not None:
            combined_input_dim += num_numerical_features

        # Weighting layers
        # Modality weights: 3 CLS embeddings
        self.text_modality_weights = nn.Parameter(torch.ones(3))  # shape (3,)
        # Per-feature categorical weights
        self.categorical_feature_weights = nn.Parameter(torch.ones(num_categorical_features))  # shape (num_categorical_features,)
        # Single numerical feature
        self.numerical_feature_weights = nn.Parameter(torch.ones(num_numerical_features)) # shape (num_numerical_features,)

        # Layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(combined_input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512) # Second hidden layer
        self.bn2 = nn.BatchNorm1d(512) # BatchNorm for the second hidden layer
        self.linear3 = nn.Linear(512, 256) # Second hidden layer
        self.bn3 = nn.BatchNorm1d(256) # BatchNorm for the second hidden layer
        self.finallinear = nn.Linear(256, 1) # Output layer


    def forward(self, categorical_inputs, numerical_inputs,
                intro_input_ids, intro_attention_mask,
                outcomes_input_ids, outcomes_attention_mask,
                criteria_input_ids, criteria_attention_mask):

        # Embed each text input using BioBERT
        intro_outputs = self.biobert(intro_input_ids, attention_mask=intro_attention_mask)
        intro_embedding = intro_outputs.pooler_output

        outcomes_outputs = self.biobert(outcomes_input_ids, attention_mask=outcomes_attention_mask)
        outcomes_embedding = outcomes_outputs.pooler_output

        criteria_outputs = self.biobert(criteria_input_ids, attention_mask=criteria_attention_mask)
        criteria_embedding = criteria_outputs.pooler_output

        ## --- Concatenate ---
        # Concatenate the embeddings from the three text inputs
        # Apply modality weights (softmax optional for normalized weight distribution)
        modality_weights = F.softmax(self.text_modality_weights, dim=0)
        text_embeddings = [
            intro_embedding * modality_weights[0],
            outcomes_embedding * modality_weights[1],
            criteria_embedding * modality_weights[2]
        ]
        text_concat = torch.cat(text_embeddings, dim=1)  # (batch, 2304)

        # Categorical: embed and apply per-feature weights
        if self.num_categorical_features is not None:
          categorical_embeds = [emb(categorical_inputs[:, i]) for i, emb in enumerate(self.categorical_embeddings)]
          categorical_embeds = [embed * self.categorical_feature_weights[i] for i, embed in enumerate(categorical_embeds)]
          cat_concat = torch.cat(categorical_embeds, dim=1)  # shape (batch, sum(emb_dims))

        # Numerical
        if self.num_numerical_features is not None:
            numerical_inputs = self.numerical_bn(numerical_inputs)  # Input numerical inputs to batch normalization layer
            # numerical_feature_weights: (num_features,) → auto-broadcast to (batch, num_features)
            weighted_numerical = numerical_inputs * self.numerical_feature_weights

        # Final concat
        combined_features = torch.cat([text_concat, cat_concat, weighted_numerical], dim=1)


        # Forward
        x = self.dropout1(combined_features)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout2(x)
        logits = self.finallinear(x)
        return logits.squeeze(-1)