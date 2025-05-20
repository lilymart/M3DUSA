import torch.nn as nn


class BilinearFusionModel(nn.Module):
    def __init__(self, embedding_dim=128, name="LF_bilinear"):
        super(BilinearFusionModel, self).__init__()
        self.name = name
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 2)

    def forward(self, e1, e2):
        # Bilinear interaction between embeddings
        e_fused = self.bilinear(e1, e2)

        logits = self.fc(e_fused)
        return logits


    def encode(self, e1, e2):
        e_fused = self.bilinear(e1, e2)
        return e_fused


"""
Overview: Bilinear models can capture interactions between two different embeddings by considering their outer product, 
thus allowing the model to learn complex relationships between them.

Explanation: Bilinear fusion models learn to combine the embeddings by considering their pairwise interactions, 
which can capture more complex dependencies between the different modalities.
'''

'''
Summary:
Weighted Fusion is simple but effective for assigning different importance to each modality.
Attention-Based Fusion allows the model to dynamically adjust the importance of each embedding based on the input.
Gated Fusion introduces a gating mechanism that controls the contribution of each embedding.
Bilinear Fusion captures complex interactions between embeddings.
Each of these methods can be a powerful alternative to simple concatenation or averaging, 
especially when the contributions of the embeddings are not equal or constant across the dataset.
"""