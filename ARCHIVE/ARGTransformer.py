import torch
import torch.nn as nn

class ARGTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embedding_dim=320, num_classes=1, num_heads=8, num_layers=2, dropout=0.1):
        super(ARGTransformer, self).__init__()
        
        # Embedding layer to replace ESM2 token representations
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_length, embedding_dim))
        
        # Define the individual layers of the transformer
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        # List to hold attention weights (we'll store them in the forward pass)
        self.attention_maps = []

    def save_attention(self, module, input, output):
        # input is a tuple, and we need the attention weights from it
        attn_weights = output[1]  # The second element in the output tuple is the attention weights
        self.attention_maps.append(attn_weights)

    def forward(self, batch_tokens, attention_mask, collect_attention=False):
        # Embedding
        x = self.embedding(batch_tokens)
        
        # Add positional encoding (truncate to sequence length)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transpose for MultiheadAttention compatibility
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, embedding_dim]
    
        # Reset attention maps for this forward pass
        self.attention_maps = []  # Ensure it's empty before starting the pass
    
        # Forward pass through each MultiheadAttention layer
        for i, attention_layer in enumerate(self.attention_layers):
            # Compute attention with need_weights=True, and average the weights
            attn_output, attn_weights = attention_layer(
                x, x, x, 
                key_padding_mask=attention_mask == 0, 
                need_weights=True,        # Request attention weights
                average_attn_weights=True # Average the attention weights across heads
            )
        
            # If collecting attention, append the attention weights directly
            if collect_attention:
                self.attention_maps.append(attn_weights)  # Append the attention weights here
            
            # Apply dropout and layer normalization
            x = self.layer_norm(attn_output + x)  # Residual connection

        # Aggregate embeddings over all tokens using mean pooling
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, embedding_dim]
        x_mean = x.mean(dim=1)  # Shape: [batch_size, embedding_dim]

        # Pass through the classification head
        logits = self.fc(x_mean)  # Shape: [batch_size, num_classes]

        # Return logits and attention maps (only if collecting attention)
        if collect_attention:
            return logits, self.attention_maps
        return logits

