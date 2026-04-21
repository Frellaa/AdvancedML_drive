import torch
import torch.nn as nn

class SP500Transformer(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=4, num_layers=2, seq_len=30, dropout=0.2):
        super(SP500Transformer, self).__init__()
        
        # 1. Project raw features to model dimension
        self.feature_projection = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding (Learned embeddings often work better than sine/cosine for finance)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True # Expects inputs as (Batch, Seq_Len, Features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Outputs a single value (logit)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Num_Features)
        
        # Project and add positional encoding
        x = self.feature_projection(x) + self.pos_encoder
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # Flatten the sequence and classify
        x = self.flatten(x)
        logits = self.classifier(x)
        
        # Squeeze to match target shape (Batch)
        return logits.squeeze()