import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_Loss(torch.nn.Module):
    def __init__(self, origin_dim, intermediate_dim, classes):
        super(Feature_Loss, self).__init__()
        self.linear = torch.nn.Linear(origin_dim, intermediate_dim)
        self.output = torch.nn.Linear(intermediate_dim, classes)

    def forward(self, x):
        x = F.relu(self.linear(x))
        y = self.output(x)
        return x, y