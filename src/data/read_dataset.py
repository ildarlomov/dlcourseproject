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

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class InferenceMCSDataset(Dataset):
    def __init__(self, tracks_df_csv, root_dir, transform=None):
        self.tracks_df = pd.read_csv(tracks_df_csv)
        # just paths to get embeddings
        self.samples = self.tracks_df.warp_path.values.tolist()
        self.root_dir = root_dir
        self.transform = transform

        print(f"Number of images for inference  is {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_image_path = self.samples[idx]

        track_image_path = os.path.join(self.root_dir, track_image_path)
        track_image = io.imread(track_image_path).astype(np.float32)
        sample = {'track_image': track_image}

        return sample


class MCSDataset(Dataset):

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


class FakeMCSDataset(Dataset):

    def __init__(self, tracks_df_csv, order_df_csv, gt_csv, root_dir, is_val=False, transform=None):
        self.samples = [(np.random.randn(112, 112, 3).astype(np.uint8),
                         np.random.randn(112, 112, 3).astype(np.uint8),
                         np.random.randn(112, 112, 3).astype(np.uint8))
                        for _ in range(100)]
        self.transform = transform
        self.tw = TransformsWrapper(transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_image, pos_image, neg_image = self.samples[idx]

        sample = {'track_image': track_image, 'pos_image': pos_image, 'neg_image': neg_image}

        if self.transform:
            sample = self.tw(sample)

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
    dataset = FakeMCSDataset(tracks_df_csv='../../data/raw/train_df.csv',
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

            print(i, sample['track_image'].size(), sample['pos_image'].size(), sample['neg_image'].size())

            if i == 3:
                break


if __name__ == '__main__':
    # example usage
    # python -i read_dataset.py check_data_iteration

    # example code
    """
    train_ds = FakeRNNMCSDataset(tracks_df_csv='data/raw/train_df.csv',
                                 order_df_csv='data/raw/train_df_track_order_df.csv',
                                 gt_csv='data/raw/train_gt_df.csv',
                                 root_dir='data/raw/data',
                                 is_val=False,
                                 transform=preprocessing)

    dev_ds = FakeRNNMCSDataset(tracks_df_csv='data/raw/train_df.csv',
                               order_df_csv='data/raw/train_df_track_order_df.csv',
                               gt_csv='data/raw/train_gt_df.csv',
                               root_dir='data/raw/data',
                               is_val=True,
                               transform=preprocessing)
    """
    check_data_iteration(iterate_data=True)
