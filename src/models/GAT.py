import torch
from torch_geometric.nn import GATv2Conv, Linear


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=128, out_channels=2, dropout=0, num_layers=2):  # hidden_channels=64
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout))
        self.lins.append(Linear(-1, hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout))
            self.lins.append(Linear(-1, hidden_channels))

        # Last layer
        self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False,
                                    dropout=dropout))  # Change out_channels to hidden_channels
        self.lins.append(Linear(-1, hidden_channels))  # Change out_channels to hidden_channels

        self.final_conv = GATv2Conv((-1, -1), out_channels, add_self_loops=False,
                                    dropout=dropout)  # Add final GAT layer for classification
        self.final_lin = Linear(-1, out_channels)  # Add final linear layer for classification

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index) + self.lins[i](x.relu())
            x = x.relu()

        # Save target_type embeddings before the final layer
        self.embeddings = self.convs[-1](x, edge_index) + self.lins[-1](x.relu())

        # Last layer (no ReLU after the last convolution)
        x = self.final_conv(self.embeddings, edge_index) + self.final_lin(self.embeddings.relu())
        return x, self.embeddings