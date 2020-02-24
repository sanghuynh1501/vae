import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature(torch.nn.Module):
    def __init__(self, origin_dim, intermediate_dim, classes):
        super(Feature, self).__init__()
        self.linear = torch.nn.Linear(origin_dim, intermediate_dim)
        self.output = torch.nn.Linear(intermediate_dim, classes)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.softmax(self.output(x))
        return x