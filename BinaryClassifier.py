# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network model
class BinaryClassifier(nn.Module):
    def __init__(self, n_samples = 5):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(10, n_samples)   # Input layer
        # Layers???
        self.fc2 = nn.Linear(n_samples, 1)    # Output layer
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(x)
        return out

