from src.models.triplet import TripletNet
from src.models.baseline_net import ResNetCaffe, BasicBlock
from src.data.read_dataset import InferenceMCSDataset
from src.data.baseline_transformers import ToTensor
from torch.utils.data.dataloader import DataLoader
import numpy as np
from src.models.predict_model import get_descriptors
from tqdm import tqdm

if __name__ == '__main__':
    weights_path = 'models/baseline/logs/6/checkpoints/best.pth'
    output_agg_path = 'models/baseline/logs/6/descriptors.npy'
    dataset = InferenceMCSDataset(tracks_df_csv='data/raw/train_df.csv',
                                  root_dir='data/raw/data',
                                  transform=ToTensor())
    model = ResNetCaffe([1, 2, 5, 3], BasicBlock, pretrained=False, weights_path=None)
    tnet = TripletNet(model, pretrained=True, weights_path=weights_path)

    get_descriptors(output_agg_path, dataset, tnet)






