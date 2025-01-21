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

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        # List to hold attention weights (we'll store them in the forward pass)
        self.attention_maps = []

    def save_attention(self, module, input, output):
        # input is a tuple, and we need the attention weights from it
        attn_weights = output[1]  # The second element in the output tuple is the attention weights
        self.attention_maps.append(attn_weights)

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

        self.num_heads = num_heads
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
        # Reset attention weights for each forward pass
        self.attention_weights = {}

        # Add batch dimension if needed
        if batch_tokens.dim() == 1:
            batch_tokens = batch_tokens.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
    
        # Embedding and positional encoding
        x = self.embedding(batch_tokens)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = x.transpose(0, 1)
    
        # Multi-head attention layers
        for i, attention_layer in enumerate(self.attention_layers):
            attn_output, attn_weights = attention_layer(
                x, x, x,
                key_padding_mask=(~attention_mask.bool()) if attention_mask.dtype != torch.bool else ~attention_mask,
                need_weights=True,
                average_attn_weights=False
            )
            if collect_attention:
                reshaped_weights = attn_weights.view(batch_tokens.size(0), self.num_heads, seq_len, seq_len)
                reshaped_weights = torch.flip(reshaped_weights, [2])
                self.attention_weights[f"layer_{i}"] = reshaped_weights
        
            x = self.layer_norm(attn_output + x)
    
        # Pooling and output
        x = x.transpose(0, 1).mean(dim=1)
        logits = self.fc(x)
        return (logits, self.attention_weights) if collect_attention else logits




