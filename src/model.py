import torch
import torch.nn as nn

class BikeDemandMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(BikeDemandMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)