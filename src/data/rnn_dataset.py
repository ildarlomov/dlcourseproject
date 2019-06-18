from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from src.data.baseline_transformers import TransformsWrapper
from tqdm import tqdm
# Ignore warnings
import warnings
import torchvision as tv

class RNNMCSDataset(Dataset):

    def __init__(self, tracks_df_csv, order_df_csv, gt_csv, root_dir, is_val=False, transform=None):
        self.order_df = pd.read_csv(order_df_csv)
        self.gt_df = pd.read_csv(gt_csv)
        self.tracks_df = pd.read_csv(tracks_df_csv)
        self.tracks_df = self.tracks_df[self.tracks_df.is_val == is_val]
        self.tw = TransformsWrapper(transform)

        self.samples = list()
        # 1 triplet sample for one person
        print(f"Generating samples for {'dev' if is_val else 'train'}")
        for person_id in tqdm(np.unique(self.tracks_df.person_id.values)):
            for track_id, person_tracks_df in self.tracks_df[self.tracks_df.person_id == person_id].groupby('track_id'):
                sampled_track_image_path = person_tracks_df.sample(1).warp_path.values[0]
                sampled_pos_image_path = self.gt_df[self.gt_df.person_id == person_id].sample(1).warp_path.values[0]
                sampled_neg_image_path = self.gt_df[self.gt_df.person_id != person_id].sample(1).warp_path.values[0]

                self.samples.append((sampled_track_image_path, sampled_pos_image_path, sampled_neg_image_path))
        self.root_dir = root_dir
        self.transform = transform

        print(f"Triplets count for {'dev' if is_val else 'train'} is {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_image_path, pos_image_path, neg_image_path = self.samples[idx]

        track_image_path = os.path.join(self.root_dir, track_image_path)
        pos_image_path = os.path.join(self.root_dir, pos_image_path)
        neg_image_path = os.path.join(self.root_dir, neg_image_path)

        track_image = io.imread(track_image_path)
        pos_image = io.imread(pos_image_path)
        neg_image = io.imread(neg_image_path)

        sample = {'track_image': track_image,
                  'pos_image': pos_image,
                  'neg_image': neg_image}

        if self.transform:
            sample = self.tw(sample)

        return sample


class FakeRNNMCSDataset(Dataset):

    def __init__(self, tracks_df_csv, order_df_csv, gt_csv, root_dir, is_val=False, transform=None):
        # [gt_image, pos_seq, neg_seq]
        seq_len = 5
        # self.samples = [[np.random.randn(112, 112, 3).astype(np.uint8),
        #                 [np.random.randn(112, 112, 3).astype(np.uint8) for i in range(seq_len)],
        #                 [np.random.randn(112, 112, 3).astype(np.uint8) for i in range(seq_len)]]
        #                 for _ in range(100)]

        self.samples = [[np.random.randn(512).astype(np.uint8),
                         [np.random.randn(512).astype(np.uint8) for i in range(seq_len)],
                         [np.random.randn(512).astype(np.uint8) for i in range(seq_len)]]
                        for _ in range(100)]

        # self.transform = transform
        # self.tw = TransformsWrapper(transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_image, pos_seq, neg_seq = self.samples[idx]
        # track_image = self.transform(track_image)
        track_image = torch.from_numpy(track_image)
        pos_seq = [torch.from_numpy(pos_img) for pos_img in pos_seq]
        neg_seq = [torch.from_numpy(neg_img) for neg_img in neg_seq]

        pos_seq = torch.stack(pos_seq, dim=0, out=None)
        neg_seq = torch.stack(neg_seq, dim=0, out=None)

        sample = {'gt_image': track_image, 'pos_seq': pos_seq, 'neg_seq': neg_seq}

        return sample


# we are going to do train dataset and test dataset separately

def check_data_iteration(iterate_data=False):
    is_val = False
    # U may use MCSDataset for the training
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = FakeRNNMCSDataset(tracks_df_csv='../../data/raw/train_df.csv',
                             order_df_csv='../../data/raw/train_df_track_order_df.csv',
                             gt_csv='../../data/raw/train_gt_df.csv',
                             root_dir='../../data/raw/data',
                             is_val=is_val,
                             transform=preprocessing)

    print(f"Total triples in {'test' if is_val else 'train'} dataset is {len(dataset)}")
    if iterate_data:
        for i in range(len(dataset)):
            sample = dataset[i]
            # print(sample['track_image'])

            print(i, sample['gt_image'].size(), sample['pos_seq'].size(), sample['neg_seq'].size())

            if i == 3:
                break


if __name__ == '__main__':
    # example usage
    # python -i read_dataset.py check_data_iteration
    check_data_iteration(iterate_data=True)
