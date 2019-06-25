from src.models.baseline_net_body import ResNetCaffeBody
import torch.nn as nn
import torch
import torch.nn.init
import math
import torch.nn.functional as F


def calculate_scale(data):
    if data.dim() == 2:
        scale = math.sqrt(3 / data.size(1))
    else:
        scale = math.sqrt(3 /
                          (data.size(1) *
                           data.size(2) *
                           data.size(3)))
    return scale


class ResNetCaffeFinetune(nn.Module):

    def __init__(self, body_model=None):
        super().__init__()
        self.body_model = body_model
        for param in self.body_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(round(12800 * self.body_model.k), 512, bias=True)

        scale = calculate_scale(self.fc.weight.data)
        torch.nn.init.uniform_(self.fc.weight.data, -scale, scale)
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()

    def forward(self, *input):
        x = self.body_model(*input)
        x = self.fc(x)
        return x
