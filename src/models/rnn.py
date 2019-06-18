from src.models.baseline_net_body import ResNetCaffeBody
import torch.nn as nn
import torch
import torch.nn.init
import math
import torch.nn.functional as F


class ResnetBasedRNN(nn.Module):

    def __init__(self, embeddingnet=None, num_layers=1, dropout=0, hidden_size=512):
        super().__init__()
        self.embeddingnet = embeddingnet
        for param in self.embeddingnet.parameters():
            param.requires_grad = False
        # requires (seq_len, bs, input_size)
        self.rlayers = nn.LSTM(batch_first=True, num_layers=num_layers, dropout=dropout, hidden_size=hidden_size)

    def forward(self, *input):
        x = self.rlayers(*input)
        return x
