import torch
import torch.nn as nn


class GatedFusionModel(nn.Module):
    def __init__(self, embedding_dim=128, name="LF_gated"):
        super(GatedFusionModel, self).__init__()
        self.name = name
        self.fc = nn.Linear(embedding_dim, 2)

        # Gating network for each embedding
        self.gate_e1 = nn.Linear(embedding_dim, embedding_dim)
        self.gate_e2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, e1, e2):
        # Compute gating values
        g_e1 = torch.sigmoid(self.gate_e1(e1))
        g_e2 = torch.sigmoid(self.gate_e2(e2))

        # Apply gating to embeddings
        e1_gated = g_e1 * e1
        e2_gated = g_e2 * e2

        # Combine the gated embeddings
        e_fused = e1_gated + e2_gated

        logits = self.fc(e_fused)
        return logits


    def encode(self, e1, e2):
        g_e1 = torch.sigmoid(self.gate_e1(e1))
        g_e2 = torch.sigmoid(self.gate_e2(e2))
        e1_gated = g_e1 * e1
        e2_gated = g_e2 * e2
        e_fused = e1_gated + e2_gated
        return e_fused


"""
Overview: Gated fusion uses gating mechanisms to control the flow of information from each embedding to the final representation. 
A gating network generates coefficients that modulate the contribution of each embedding

Explanation: The gating mechanism allows each embedding to be selectively suppressed or amplified. 
The gates control the flow of information from each embedding to the final representation, 
effectively learning which parts of each embedding are more relevant for the task.
"""