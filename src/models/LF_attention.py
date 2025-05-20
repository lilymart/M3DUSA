import torch
import torch.nn as nn


class AttentionFusionModel(nn.Module):
    def __init__(self, embedding_dim=128, name ="LF_attention"):
        super(AttentionFusionModel, self).__init__()
        self.name = name
        self.fc = nn.Linear(embedding_dim, 2)

        # Attention layers for each embedding
        self.attention_e1 = nn.Linear(embedding_dim, 1)
        self.attention_e2 = nn.Linear(embedding_dim, 1)

    def forward(self, e1, e2):
        # Compute attention scores
        attn_score_e1 = torch.sigmoid(self.attention_e1(e1))
        attn_score_e2 = torch.sigmoid(self.attention_e2(e2))

        # Normalize attention scores
        attn_sum = attn_score_e1 + attn_score_e2
        attn_score_e1 = attn_score_e1 / attn_sum
        attn_score_e2 = attn_score_e2 / attn_sum

        # Weighted combination based on attention scores
        e_fused = attn_score_e1 * e1 + attn_score_e2 * e2

        logits = self.fc(e_fused)
        return logits


    def encode(self, e1, e2):
        attn_score_e1 = torch.sigmoid(self.attention_e1(e1))
        attn_score_e2 = torch.sigmoid(self.attention_e2(e2))
        attn_sum = attn_score_e1 + attn_score_e2
        attn_score_e1 = attn_score_e1 / attn_sum
        attn_score_e2 = attn_score_e2 / attn_sum
        e_fused = attn_score_e1 * e1 + attn_score_e2 * e2
        return e_fused


"""
Overview: Attention mechanisms allow the model to dynamically adjust the contribution of each embedding depending on the input. 
This is particularly useful when the importance of each embedding may vary across different samples.

Explanation: attention scores are learned for each embedding. These scores are dynamically computed for each input, 
allowing the model to adjust how much weight to assign to each embedding based on the input features.
"""