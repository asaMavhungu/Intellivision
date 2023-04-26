from utils import *

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, padding=0) # First Conv Layer
        self.pool = nn.MaxPool2d(2)  # For pooling
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(5*16*16, 120)  # First FC HL
        self.fc2= nn.Linear(120, 10) # Output layer

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.conv1(x)) # Shape: (B, 5, 32, 32)
      x = self.pool(x)  # Shape: (B, 5, 16, 16)
      x = self.flatten(x) # Shape: (B, 980)
      x = F.relu(self.fc1(x))  # Shape (B, 256)
      x = self.fc2(x)  # Shape: (B, 10)
      return x  