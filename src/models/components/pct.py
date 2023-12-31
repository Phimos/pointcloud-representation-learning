import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear as Lin
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_geometric.utils import scatter

# path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data/ModelNet10")
# pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
# train_dataset = ModelNet(path, "10", True, transform, pre_transform)
# test_dataset = ModelNet(path, "10", False, transform, pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce cardinality and uses an mlp to augment features
    dimensionnality."""

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
            reduce="max",
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k)
            )

            self.transformers_down.append(TransformerBlock(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndim == 3 and data.shape[1] == 3
        batch_size, _, num_points = data.shape
        pos = torch.einsum("b d n -> b n d", data).reshape(batch_size * num_points, 3)
        batch = torch.arange(batch_size, device=data.device).repeat_interleave(num_points)
        x = torch.ones((batch_size * num_points, 1), device=data.device)

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)
        return x

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self.encode(data)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)
