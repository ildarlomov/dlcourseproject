import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):
    def __init__(self, embeddingnet, pretrained=False, weights_path=None):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet
        if pretrained:
            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'
            weights = torch.load(weights_path, map_location=map_location)
            self.load_state_dict(weights["model_state_dict"])

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

class RNNTripletNet(nn.Module):
    def __init__(self, embeddingnet: nn.LSTM, pretrained=False, weights_path=None):
        super(RNNTripletNet, self).__init__()
        self.embeddingnet = embeddingnet
        if pretrained:
            if torch.cuda.is_available():
                # map_location = lambda storage, loc: storage.cuda()
                map_location = 'cpu'
            else:
                map_location = 'cpu'
            weights = torch.load(weights_path, map_location=map_location)
            self.load_state_dict(weights["model_state_dict"])

    def forward(self, x, y, z):
        output_y, (hn_y, cn_y) = self.embeddingnet(y)
        output_z, (hn_z, cn_z) = self.embeddingnet(z)
        hn_y = torch.squeeze(hn_y)
        hn_z = torch.squeeze(hn_z)
        dist_a = F.pairwise_distance(x, hn_y, 2)
        dist_b = F.pairwise_distance(x, hn_z, 2)
        return dist_a, dist_b, x, hn_y, hn_z


