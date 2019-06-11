from src.models.triplet import TripletNet
from src.models.baseline_net import ResNetCaffe, BasicBlock
from src.data.read_dataset import InferenceMCSDataset
from src.data.baseline_transformers import ToTensor
from torch.utils.data.dataloader import DataLoader
import numpy as np
from src.models.predict_model import get_descriptors
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('test_df_path', type=str, help='Path to test data images')
    parser.add_argument('weights_path', type=str, help='Path to weights')
    parser.add_argument('output_desc_path', type=str, help='Path to test_descriptors.npy')
    parser.add_argument('root_dir', type=str, help='Path to dataset')
    conf = parser.parse_args()

    weights_path = conf.weights_path
    output_agg_path = conf.output_desc_path
    dataset = InferenceMCSDataset(tracks_df_csv=conf.test_df_path,
                                  root_dir=conf.root_dir,
                                  transform=ToTensor())
    model = ResNetCaffe([1, 2, 5, 3], BasicBlock, pretrained=False, weights_path=None)
    tnet = TripletNet(model, pretrained=True, weights_path=weights_path)

    get_descriptors(output_agg_path, dataset, tnet)
