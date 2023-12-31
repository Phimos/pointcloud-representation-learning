import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, channels: int, num_points: int):
        super().__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(channels, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_points * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(batch_size, 3, self.num_points)
        return x
