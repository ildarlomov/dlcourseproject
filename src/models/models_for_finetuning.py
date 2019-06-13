from src.models.baseline_net_body import ResNetCaffeBody
import torch.nn as nn
import torch
import torch.nn.init
import math
import torch.nn.functional as F


def set_false_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class ResNetCaffeFinetune(nn.Module):

    def __init__(self, body_model=None):
        super().__init__()
        self.body_model = body_model
        for param in self.body_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(round(12800 * self.body_model.k), 512, bias=True)

    def forward(self, *input):
        x = self.body_model(*input)
        x = self.fc(x)
        return x
