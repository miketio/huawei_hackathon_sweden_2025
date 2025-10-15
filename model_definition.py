# model_definition.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.profiler import profile, ProfilerActivity


class SVDNet(nn.Module):
    def __init__(self, M=128, N=128, k=64):
        """
        Updated for Round 2: 128x128 matrices, rank 64 approximation
        """
        super(SVDNet, self).__init__()
        self.M = M
        self.N = N
        self.k = k
        
        # === 1. CNN Encoder (Updated for 128x128 input) ===
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ELU(),
            # nn.Dropout(0.05),
            
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ELU(),
            # nn.Dropout(0.05),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ELU(),
            # nn.Dropout(0.05),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ELU(),
            # nn.Dropout(0.05)
            
        )
        
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # === 2. Transformer Decoder ===
        d_model = 16 * 8 * 8  # 2048
        
        # Optimal dimension/head ratio between 40 and 50
        num_heads = 16  # 2048 / 40 = 51.2
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,  # 8192
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # === 3. Output Heads (Updated for rank 64) ===
        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ELU(),
            nn.Linear(64, k)
        )
        
        self.U_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ELU(),
            nn.Linear(256, k * M * 2)
        )
        
        self.V_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ELU(),
            nn.Linear(256, k * N * 2)
        )
        
        # Initialize weights to avoid zero solution
        self._initialize_weights()

    def _initialize_weights(self):
        """Proper initialization to avoid all-zeros local minimum"""
        # Initialize sigma head to produce positive values
        nn.init.kaiming_normal_(self.sigma_head[2].weight)
        nn.init.constant_(self.sigma_head[2].bias, 1.0)  # Start with ones
        
        # Initialize U and V heads
        nn.init.kaiming_normal_(self.U_head[2].weight)
        nn.init.kaiming_normal_(self.V_head[2].weight)
        nn.init.zeros_(self.U_head[2].bias)
        nn.init.zeros_(self.V_head[2].bias)

    def forward(self, x):
        """
        Forward pass for 128x128 matrices and rank 64 approximation
        """
        # x: [B, 2, 128, 128]
        x = self.encoder(x)
        x = self.pool(x)
        
        # Flatten for transformer [B, 1, d_model]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, -1)
        
        # Apply transformer
        x = self.transformer_layer(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [B, d_model]
        
        # Generate singular values (always positive)
        sigma = F.softplus(self.sigma_head(x))  # [B, k]
        
        # Prevent sigma from becoming too small
        sigma = torch.clamp(sigma, min=1e-4)  
        
        # Generate U and V with proper dimensions
        U_flat = self.U_head(x).view(-1, self.M, self.k, 2)  # [B, M, k, 2]
        V_flat = self.V_head(x).view(-1, self.N, self.k, 2)  # [B, N, k, 2]
        
        # === Normalize U and V ===
        # U normalization
        U_norm = torch.norm(U_flat, p=2, dim=1, keepdim=True) + 1e-8
        U_norm = torch.norm(U_norm, p=2, dim=3, keepdim=True) + 1e-8
        U_normalized = U_flat / U_norm
        
        # V normalization
        V_norm = torch.norm(V_flat, p=2, dim=1, keepdim=True) + 1e-8
        V_norm = torch.norm(V_norm, p=2, dim=3, keepdim=True) + 1e-8
        V_normalized = V_flat / V_norm
        
        return sigma, U_normalized, V_normalized

def get_avg_flops(model:nn.Module, input_data:Tensor)->float:
    """
    Estimates the average FLOPs per sample for a model using PyTorch Profiler.
    
    Args:
        model (torch.nn.Module): The neural network model to profile.
        input_data (torch.Tensor): Input tensor for the model (must include batch dimension).
    
    Returns:
        float: Average Mega FLOPs per sample in the batch.
    
    Raises:
        RuntimeError: If no CUDA device is available or input batch size is 0.
    """
    
    # Ensure batch dimension exists
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")
    
    batch_size = input_data.size(0)
    
    # Evaluation mode, improved inference and freeze norm layers
    model = model.eval().cpu()
    input_data = input_data.cpu()
    
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False
        ) as prof:
            model(input_data)
    # Calculate total FLOPs
    total_flops = sum(event.flops for event in prof.events())
    avg_flops = total_flops / batch_size
    
    return avg_flops * 1e-6 / 2


## Example usage 

if __name__ == "__main__":
    model = SVDNet()
    model.load_state_dict(torch.load("model_weights.pth", map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully from 'model_weights.pth'")
    # Create a dummy input matching your input format [1, 2, 128, 128]
    dummy_input = torch.randn(1, 2, 128, 128)
    C = get_avg_flops(model, dummy_input)
    print(f"Measured: {C:.2f} Mega MACs")