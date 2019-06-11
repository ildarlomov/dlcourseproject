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


