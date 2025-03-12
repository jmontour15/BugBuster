import torch
from torch import nn
from esm import pretrained
from ClassificationHead import ClassificationHead
from tqdm import tqdm
import peft
from peft import LoraConfig, get_peft_model

class BugBuster(nn.Module):
    def __init__(self, num_mlm_layers: int, num_classes=None, num_heads: int=8, num_layers: int=1, dropout: float=0.2, freeze_mlm: bool=True, LoRA: bool=False):
        super().__init__()

        # Load pre-trained ESM model based on num_mlm_layers
        esm_model, alphabet = self.load_esm_model(num_mlm_layers)

        # LoRA requires the MLM model to be frozen
        if LoRA and not freeze_mlm:
            raise ValueError("Cannot unfreeze ESM and use LoRA. If you want to use LoRA, set `freeze_mlm=True`.")

        if LoRA:
            esm_model = self.apply_lora(esm_model)  # Apply LoRA to the model

        if not freeze_mlm:
            for param in esm_model.parameters():
                param.requires_grad = True

        self.num_mlm_layers = num_mlm_layers
        
        # Store model and classification head
        self.ProteinMLM = esm_model
        self.TaskSpecificHead = ClassificationHead(esm_model.embed_tokens.embedding_dim, num_classes, num_heads, int(num_layers), dropout)
        self.batch_converter = alphabet.get_batch_converter()

    def load_esm_model(self, num_mlm_layers):
        """ Loads the pre-trained ESM model based on the specified layer count. """
        model_map = {
            6: pretrained.esm2_t6_8M_UR50D,
            12: pretrained.esm2_t12_35M_UR50D,
            30: pretrained.esm2_t30_150M_UR50D,
            33: pretrained.esm2_t33_650M_UR50D,
            36: pretrained.esm2_t36_3B_UR50D,
            48: pretrained.esm2_t48_15B_UR50D
        }
        if num_mlm_layers not in model_map:
            raise ValueError(f'Invalid number of ESM layers: {num_mlm_layers}. Must be one of {list(model_map.keys())}.')
        return model_map[num_mlm_layers]()

    def apply_lora(self, model):
        """ Applies LoRA to the self-attention layers of the model. """
        config = LoraConfig(
            r=16,  # Rank of the LoRA update matrices
            lora_alpha=32,  # Scaling factor for LoRA
            lora_dropout=0.1,
            bias="none",
            target_modules=["self_attn.v_proj", "self_attn.q_proj", "self_attn.k_proj"]
        )
        return get_peft_model(model, config)

    def get_attention_weights(self, batch_tokens, attention_masks, need_head_weights=False):
        """
        params:
            batch_tokens: tensor containing tokens from batch whos attention weights will be returned
             - shape: [batch_size, max_sequence_length]
            attention_masks: tensor containing a binary (0/1) denoting padding tokens, where 1 represents a real token and 0 represents padding
             - shape: [batch_size, max_sequence_length]
        returns:
            weights: tensor of attention weights
             - If need_head_weights == False, tensor shape will be [batch_size, num_layers, sequence_length, sequence_length]
             - If need_head_weights == True, tensor shape will be [batch_size, num_layers, num_heads, sequence_length, sequence_length]
        """
        # Initialize weights dictionary
        weights = {}
        
        # Forward pass with `need_head_weights=True` to get attention weights
        with torch.no_grad():
            output = self.ProteinMLM(batch_tokens, repr_layers=[self.num_mlm_layers], need_head_weights=True) # Need to set this to true to access weights
            token_representations = output["representations"][self.num_mlm_layers] # divide by 3 in order to access low-level embeddings
        # Extract attention weights
        MLM_weights = output["attentions"]
        # Remove erroneous first dimension
        MLM_weights = MLM_weights.squeeze(0)

        if need_head_weights:
            # Returns un-aggregated weights
            weights["MLM_weights"] = MLM_weights
        else:
            # Aggregate attention over heads by mean
            MLM_weights = torch.mean(MLM_weights, dim=2)
            weights["MLM_weights"] = MLM_weights

        print(token_representations.shape)
        task_specific_head_weights = self.TaskSpecificHead.get_attention_weights(token_representations, attention_masks, need_head_weights)
        weights["task_specific_head_weights"] = task_specific_head_weights

        return weights

    def train_step(self, train_loader, loss_fn, optimizer, device):
        """
        Runs one training epoch and returns the average training loss.
        """
        self.train()
        train_loss = 0.0

        for sequences, attention_masks, targets in tqdm(train_loader, desc="Training"):
            # Send data to device
            sequences = sequences.to(device)
            attention_masks = attention_masks.to(device)
            targets = targets.to(device).float().unsqueeze(1)  # Shape: [batch_size, 1]

            # Forward pass
            optimizer.zero_grad()
            outputs = self(sequences, attention_mask=attention_masks)

            # Compute the loss
            loss = loss_fn(outputs, targets)
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

        return train_loss / len(train_loader)  # Return average loss

    def test_step(self, val_loader, loss_fn, device):
        """
        Runs one validation/testing epoch and returns the average validation loss.
        """
        self.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sequences, attention_masks, targets in tqdm(val_loader, desc="Validating"):
                sequences = sequences.to(device)
                attention_masks = attention_masks.to(device)
                targets = targets.to(device).float().unsqueeze(1) # float().unsqueeze(1)

                outputs = self(sequences, attention_mask=attention_masks)

                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)  # Return average validation loss

    def forward(self, batch_tokens, attention_mask, freeze_MLM = True):
        # Generate token representations
        if freeze_MLM: # Freeze weights (no fine-tuning)
            with torch.no_grad():
                results = self.ProteinMLM(batch_tokens, repr_layers=[self.num_mlm_layers])
                token_representations = results["representations"][self.num_mlm_layers]  # Shape: [batch_size, seq_len, embedding_dim]
        else: # Unfreeze weights (allow for fine-tuning)
            results = self.ProteinMLM(batch_tokens, repr_layers=[self.num_mlm_layers])
            token_representations = results["representations"][self.num_mlm_layers]  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Classify resulting token representations
        prediction = self.TaskSpecificHead(token_representations, attention_mask)

        return prediction
        