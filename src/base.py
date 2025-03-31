import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(10):  # 10 layers
            layers.append(nn.Linear(10000, 10000))  # 10K nodes
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.resource_gov = lambda x: x * 0.05  # 5% active

    def forward(self, x):
        return self.resource_gov(self.net(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(10000, 10000), nn.ReLU(),
            nn.Linear(10000, 10000)
        )
        self.skip = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)  # Skip connection

# Init
mlp = MLP().to('cpu')
resnet = ResNet().to('cuda')
input_tensor = torch.randn(1, 10000)  # Batch 1, 10K features