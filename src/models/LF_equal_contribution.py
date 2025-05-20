import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    def __init__(self, embedding_dim=128, fusion_strategy='concat', name="LF_concat"):
        super(LateFusionModel, self).__init__()

        self.fusion_strategy = fusion_strategy
        self.name = name

        if fusion_strategy == 'concat':
            self.fc = nn.Linear(2 * embedding_dim, 2)  # For concatenation
        elif fusion_strategy == 'avg':
            self.fc = nn.Linear(embedding_dim, 2)      # For average pooling
        elif fusion_strategy == 'max':
            self.fc = nn.Linear(embedding_dim, 2)      # For max pooling
        else:
            raise ValueError("Invalid fusion strategy. Choose from ['concat', 'avg', 'max']")


    def forward(self, e1, e2):
        if self.fusion_strategy == 'concat':
            e_fused = torch.cat((e1, e2), dim=-1)  # Concatenate along the feature dimension
        elif self.fusion_strategy == 'avg':
            e_fused = (e1 + e2) / 2  # Average pooling
        elif self.fusion_strategy == 'max':
            e_fused = torch.max(e1, e2)  # Max pooling

        logits = self.fc(e_fused)  # Linear layer output
        return logits

    def encode(self, e1, e2):
        """
        Return the fused embedding (without applying the final classification layer).
        In this case, for concatenation we simply return the concatenated vector.
        """
        if self.fusion_strategy == 'concat':
            fused = torch.cat((e1, e2), dim=-1)
        elif self.fusion_strategy == 'avg':
            fused = (e1 + e2) / 2
        elif self.fusion_strategy == 'max':
            fused = torch.max(e1, e2)
        return fused