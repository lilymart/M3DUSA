import torch.nn as nn

class EnhancedClassifier(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Define individual layers
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim // 4, embedding_dim // 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(embedding_dim // 8, 2) # 2 output neurons
        #self.fc3 = nn.Linear(embedding_dim // 8, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through all layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x) # Output raw logits
        return x #return self.sigmoid(x)

    def encode(self, x):
        # Pass through layers up to the one before the classification layer
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return self.relu2(x)