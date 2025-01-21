import torch
import torch.nn as nn
import esm

class ESM2ClassifierWithAttention(nn.Module):
    def __init__(self, esm_model, embedding_dim=320, num_classes=1, num_heads=8, num_layers=2, dropout=0.1):
        super(ESM2ClassifierWithAttention, self).__init__()
        self.esm_model = esm_model
        
        # Store ESM2 attention modules for easier access
        self.esm_attention_layers = nn.ModuleList([
            layer.attention for layer in self.esm_model.layers
        ])
        
        # Custom attention layers
        self.custom_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        # Single list to store aggregated attention weights
        self.attention_maps = []
        
    def forward(self, batch_tokens, attention_mask, collect_attention=False):
        # Reset attention maps
        self.attention_maps = []
        
        # Get the token representations from the ESM model
        with torch.no_grad():
            # Collect ESM attention weights during forward pass
            all_attention_weights = []
            
            # Modified ESM forward pass to collect attention weights
            token_embeddings = self.esm_model.embed_tokens(batch_tokens)
            x = self.esm_model.layers[0].input_layernorm(token_embeddings)
            
            # Process through ESM layers and collect attention
            for layer in self.esm_model.layers:
                # Get attention weights from ESM layer
                attn_output, attn_weights = layer.attention(
                    x,
                    x,
                    x,
                    need_weights=True,
                    average_attn_weights=True  # Average across heads
                )
                
                if collect_attention:
                    self.attention_maps.append(attn_weights)
                
                # Continue with the rest of the layer processing
                x = layer.attention_layer_norm(x + attn_output)
                m = layer.ffn_layer_norm(x + layer.feed_forward(x))
                x = m
            
            # Get final token representations
            token_representations = x
        
        # Transpose for MultiheadAttention compatibility
        x = token_representations.transpose(0, 1)
        
        # Forward pass through custom attention layers
        for attention_layer in self.custom_attention_layers:
            attn_output, attn_weights = attention_layer(
                x, x, x,
                key_padding_mask=attention_mask == 0,
                need_weights=True,
                average_attn_weights=True
            )
            
            if collect_attention:
                self.attention_maps.append(attn_weights)
            
            x = self.layer_norm(attn_output + x)
        
        # Aggregate embeddings
        x_mean = x.mean(dim=0)
        logits = self.fc(x_mean)
        
        if collect_attention:
            # Stack all attention maps along a new dimension
            # Each element in attention_maps should be of shape (batch_size, seq_len, seq_len)
            stacked_attention = torch.stack(self.attention_maps, dim=0)  # (num_layers, batch_size, seq_len, seq_len)
            
            # Calculate mean attention across all layers
            mean_attention = stacked_attention.mean(dim=0)  # (batch_size, seq_len, seq_len)
            
            return logits, {
                'per_layer_attention': self.attention_maps,  # List of attention maps from each layer
                'mean_attention': mean_attention,  # Average attention across all layers
                'stacked_attention': stacked_attention  # All attention maps stacked together
            }
            
        return logits

# Helper function to analyze attention patterns
def analyze_attention_patterns(attention_maps):
    per_layer_attention = attention_maps['per_layer_attention']
    mean_attention = attention_maps['mean_attention']
    stacked_attention = attention_maps['stacked_attention']
    
    num_layers = len(per_layer_attention)
    batch_size, seq_len, _ = mean_attention.shape
    
    print(f"Number of attention layers: {num_layers}")
    print(f"Shape of mean attention: {mean_attention.shape}")
    print(f"Shape of stacked attention: {stacked_attention.shape}")
    
    # Print per-layer attention shapes
    for i, attn in enumerate(per_layer_attention):
        print(f"Layer {i} attention shape: {attn.shape}")
    
    return {
        'num_layers': num_layers,
        'sequence_length': seq_len,
        'batch_size': batch_size
    }

# Usage example
def collect_and_analyze_attention(model, test_loader, device):
    for batch_idx, (batch_tokens, attention_mask, labels) in enumerate(test_loader):
        if batch_idx == 0:
            batch_tokens = batch_tokens.to(device)
            attention_masks = attention_mask.to(device)
            
            model.eval()
            with torch.no_grad():
                logits, attention_maps = model(batch_tokens, attention_masks, collect_attention=True)
                
            # Analyze attention patterns
            stats = analyze_attention_patterns(attention_maps)
            
            return attention_maps, stats
            
        break

'''
I'll help modify the model to aggregate attention maps across all layers. You're correct that the attention weights will typically have the shape (batch_size, sequence_len, sequence_len) when averaged across heads. Let's modify the code to handle this aggregation.
ESM2 Classifier with Aggregated Attention CollectionClick to open code
Key changes I've made:

Simplified the attention collection into a single list self.attention_maps
Modified the ESM forward pass to collect attention weights directly from each layer
Added aggregation of attention maps:

per_layer_attention: List of individual attention maps from each layer
mean_attention: Average attention across all layers
stacked_attention: All attention maps stacked into a single tensor


Added helper functions to analyze the attention patterns

The attention weights will have these shapes:

Per-layer attention: (batch_size, sequence_len, sequence_len)
Stacked attention: (num_layers, batch_size, sequence_len, sequence_len)
Mean attention: (batch_size, sequence_len, sequence_len)

To use this modified version:

# Initialize and use the model
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = ESM2ClassifierWithAttention(esm_model)
model = model.to(device)

# Collect and analyze attention
attention_maps, stats = collect_and_analyze_attention(model, test_loader, device)

# Access different types of attention
per_layer = attention_maps['per_layer_attention']
mean_attn = attention_maps['mean_attention']
stacked_attn = attention_maps['stacked_attention']
'''