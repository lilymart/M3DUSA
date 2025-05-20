import torch
import torch.nn as nn


class WeightedFusionModel(nn.Module):
    def __init__(self, embedding_dim=128, name="LF_weighted"):
        super(WeightedFusionModel, self).__init__()
        self.name = name
        self.fc = nn.Linear(embedding_dim, 2)
        self.weight1 = nn.Parameter(torch.tensor(0.5))  # Learnable weight for e_net
        self.weight2 = nn.Parameter(torch.tensor(0.5))  # Learnable weight for e_text


    def forward(self, e1, e2):
        # Normalize the weights so they sum to 1
        weight1 = torch.sigmoid(self.weight1)
        weight2 = torch.sigmoid(self.weight2)

        # Weighted combination
        e_fused = weight1 * e1 + weight2 * e2

        logits = self.fc(e_fused)
        return logits

    """
    Compute and return the fused embedding from e1 and e2.
    This is done using the same weighted combination as in the forward pass,
    but without applying the final classification layer.
    """
    def encode(self, e1, e2):
        weight1 = torch.sigmoid(self.weight1)
        weight2 = torch.sigmoid(self.weight2)
        e_fused = weight1 * e1 + weight2 * e2
        return e_fused



"""
Overview: This approach learns a scalar weight for each embedding, allowing the model to give more importance to 
one embedding over the other. The weights are learned during training.

Explanation: The model learns weights weight1 and weight2 during training. 
These weights control how much each embedding contributes to the final decision. 
The sigmoid function ensures that the weights are in the range [0, 1].
"""