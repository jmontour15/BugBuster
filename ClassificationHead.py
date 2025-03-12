import torch
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim:int, num_classes:int, num_heads: int=6, num_layers: int=1, dropout: float=0.1):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f'The embedding dimension of {embedding_dim} is not divisible by the provided num heads ({num_heads}). Please change num_heads to an integer that cleanly divides {embedding_dim}.')
        # Attention layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for layer in range(num_layers)
        ])

        # Performance increasing layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # Final classification head
        self.fc = nn.Linear(embedding_dim, num_classes)
        # Final output
        self.sigmoid = nn.Sigmoid()
    
    def get_attention_weights(self, tokens, attention_mask, need_head_weights=False):
        attn_maps = []

        for attn_layer in self.attn_layers:
            _, attn_weights = attn_layer(tokens, tokens, tokens,
                                         key_padding_mask=attention_mask == 0,
                                         need_weights=True,
                                         average_attn_weights = False)
            attn_maps.append(attn_weights)  # Shape: [batch_size, num_heads, seq_len, seq_len]
        
        attn_maps = torch.stack(attn_maps, dim=1)  # Shape: [batch_size, num_layers, num_heads, seq_len, seq_len]
        
        if not need_head_weights:
            attn_maps = attn_maps.mean(dim=2)  # Aggregate over heads, shape: [batch_size, num_layers, seq_len, seq_len]

        return attn_maps

    def forward(self, token_representations, attention_mask):
        # Start with input embeddings
        x = token_representations
    
        # Pass through attention layer(s)
        for attn_layer in self.attn_layers:
            # Apply attention
            attn_output, _ = attn_layer(x, x, x, key_padding_mask=attention_mask==0)
            # Add skip connection and normalize
            x = self.layer_norm(x + attn_output)  # Proper residual connection
    
        # Aggregate features (mean pooling)
        pooled_output = x.mean(dim=1)
        # Linear layer
        logits = self.fc(pooled_output)
        # Sigmoid
        # output = self.sigmoid(logits)
        return logits
    

        
        