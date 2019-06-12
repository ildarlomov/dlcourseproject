import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import torchvision as tv
import torch
import torch.nn.functional as F

from src.models.resnet_caffe import ResNetCaffe
from dataset import ImageListDataset


def main(conf):
    model = ResNetCaffe([1, 2, 5, 3], pretrained=True, weights_file=conf.weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD)])

    df = pd.read_csv(conf.df_path)
    dataset = ImageListDataset(data_root=conf.root_path,
                               data_list=df.warp_path.values,
                               transform=preprocessing)

    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=256,
                                                 shuffle=False,
                                                 num_workers=6,
                                                 pin_memory=True,
                                                 drop_last=False)

    result_arr = np.empty((0, 512), dtype=np.float32)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
            data = data.to(device)
            output = model(data)
            output = F.normalize(output, dim=1)
            output = output.detach().cpu().numpy()
            result_arr = np.vstack((result_arr, output))

    np.save(conf.descriptors_path, result_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('--root_path',
                        type=str,
                        help='Path to train data directory')
    parser.add_argument('--df_path',
                        type=str,
                        help='Path to csv file with warp_path column (train_df.csv or train_gt_df.csv)')
    parser.add_argument('--descriptors_path',
                        type=str,
                        help='Path to descriptors.npy file')
    parser.add_argument('--weights_path',
                        type=str,
                        help='Path to model weights file')
    conf = parser.parse_args()
    main(conf)
