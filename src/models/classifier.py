import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(SimpleClassifier, self).__init__()
        # Define a linear layer to map the embedding to class scores
        self.fc = nn.Linear(embedding_dim, 2)

    def forward(self, x):
        # Forward pass through the linear layer
        logits = self.fc(x)
        return logits