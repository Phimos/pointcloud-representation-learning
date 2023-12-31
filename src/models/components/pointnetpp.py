import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius


class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 3 and data.shape[1] == 3
        batch_size, _, num_points = data.shape
        pos = torch.einsum("b d n -> b n d", data).reshape(batch_size * num_points, 3)
        batch = torch.arange(batch_size, device=data.device).repeat_interleave(num_points)
        sa0_out = (None, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out
        return x

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            data (torch.Tensor): Input data point cloud of shape (B, 3, N).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        x = self.encode(data)
        return self.mlp(x).log_softmax(dim=-1)
