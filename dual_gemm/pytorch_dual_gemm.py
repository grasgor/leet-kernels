'''
    Simple dual GEMM with silu activation,
    C = silu(A @ B1) * (A @ B2).
    A = (N x K)
    B1 = (K x M1)
    B2 = (K x M2) --> M1 + M2 = 2M and B1 plus B2 (not addition) would make a weight matrix of shape = (K x 2M)
    C = N x M
'''

import torch
import torch.nn as nn

class DualGEMM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x):
        # let X1 = x @ W1 and X2 = x @ W2
        # w = [W1, W2]
        x = self.linear(x)
        X1, X2 = x.chunk(2, dim=-1)  # shape: each (batch_size, output_dim)
        
        # Apply SiLU gating
        return nn.functional.silu(X1) * X2
    

class DualGEMM_scratch():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_weight_matrices()

    def silu(self, x):
        return x / (1 + torch.exp(-x))

    def init_weight_matrices(self):
        self.W = torch.rand((self.input_dim, 2 * self.output_dim))
        self.W1, self.W2 = (self.W).chunk(2, dim = -1) #split along column
        
    def forward(self, X):
        self.X1 = self.silu(X @ self.W1)
        self.X2 = X @ self.W2
        return torch.mul(self.X1, self.X2)
    

if __name__ == "__main__":
    batch_dim = 8
    input_dim = 512
    output_dim = 128

    X = torch.rand((batch_dim, input_dim))
    module = DualGEMM(input_dim, output_dim)
    C_module = module(X)
    print(f"PyTorch Module output shape: {C_module.shape}")

    # Scratch version
    scratch = DualGEMM_scratch(input_dim, output_dim)
    C_scratch = scratch.forward(X)
    print(f"Scratch implementation output shape: {C_scratch.shape}")

