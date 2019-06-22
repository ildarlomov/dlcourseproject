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
import pickle


class FinalRNNMCSDataset(Dataset):

    def __init__(
            self,
            test_df: str,
            test_df_track_order_df,
            test_descriptors_df,
            root_dir,
            transform=None
    ):
        """ Plan
            1. for each track presented form 1 triplet with each of gt images vs one random negative track
            and lets work with indices only
        """
        self.test_df = pd.read_csv(test_df)
        self.test_df_track_order_df = pd.read_csv(test_df_track_order_df)
        self.test_descriptors_npy = np.load(test_descriptors_df)
        self.samples = list()
        # 1 triplet sample for one person
        # this takes 10 minutes every run
        print('Generating dataset for evaluation')
        for id, (track_id) in tqdm(self.test_df_track_order_df[['track_id']].iterrows(), total=len(self.test_df_track_order_df)):
            track_image_idxs = self.test_df[self.test_df.track_id == track_id].index.values
            self.samples.append((track_image_idxs))
        self.root_dir = root_dir
        self.transform = transform

        print(f"Triplets count for final eval is {len(self.samples)}")
        # Was Triplets count for train was 57570 when only one negative sample was used
        # now it s 1151400 (20 times more)
        # with open('train_samples.pkl', 'wb') as outf:
        #     pickle.dump(self.samples, outf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pos_images_idxs = self.samples[idx]
        # todo: maybe add some scaling on all given descriptors
        pos_seq = self.test_descriptors_npy[pos_images_idxs]

        pos_seq = [torch.from_numpy(pos_img) for pos_img in pos_seq]

        pos_seq = torch.stack(pos_seq, dim=0, out=None)

        sample = {'img_seq': pos_seq}

        return sample

class RNNMCSDataset(Dataset):

    def __init__(
            self,
            train_df: str,
            train_df_descriptors,
            train_gt_df,
            train_gt_descriptors,
            train_df_track_order_df,
            root_dir,
            is_val=False,
            transform=None
    ):
        """ Plan
            1. for each track presented form 1 triplet with each of gt images vs one random negative track
            and lets work with indices only
        """
        self.train_df = pd.read_csv(train_df)
        self.train_df_descriptors = np.load(train_df_descriptors)
        self.train_gt_df = pd.read_csv(train_gt_df)
        self.train_gt_descriptors = np.load(train_gt_descriptors)
        self.train_df_track_order_df = pd.read_csv(train_df_track_order_df)
        self.train_df_track_order_df = pd.merge(self.train_df_track_order_df,
                                                self.train_df[['person_id', 'is_val']].drop_duplicates(),
                                                on='person_id',
                                                how='left')  # [is_val == False]
        self.train_df_track_order_df = self.train_df_track_order_df[self.train_df_track_order_df.is_val == is_val]

        self.samples = list()
        # 1 triplet sample for one person
        # this takes 10 minutes every run
        if is_val:
            n_neg_samples = 1
            print(f"Generating samples for {'dev' if is_val else 'train'}")
            for id, (track_id, person_id) in tqdm(self.train_df_track_order_df[['track_id', 'person_id']].iterrows(), total=len(self.train_df_track_order_df)):
                not_this_person_order_df = self.train_df_track_order_df[self.train_df_track_order_df.person_id != person_id]
                track_image_idxs = self.train_df[self.train_df.track_id == track_id].index.values
                track_anchors_df = self.train_gt_df[self.train_gt_df.person_id == person_id]
                for anchor_idx in track_anchors_df.index.values:
                    for not_this_person_sampled_track_id in tqdm(not_this_person_order_df.sample(n_neg_samples).track_id.values):
                        not_this_person_sampled_track_image_idxs = self.train_df[
                            self.train_df.track_id == not_this_person_sampled_track_id].index.values

                        self.samples.append((anchor_idx, track_image_idxs, not_this_person_sampled_track_image_idxs))
                # if id > 10:
                #     break


        else:

            with open('train_samples.pkl', 'rb') as inf:
                self.samples = pickle.loads(inf.read())
        self.root_dir = root_dir
        self.transform = transform

        print(f"Triplets count for {'dev' if is_val else 'train'} is {len(self.samples)}")
        # Was Triplets count for train was 57570 when only one negative sample was used
        # now it s 1151400 (20 times more)
        # with open('train_samples.pkl', 'wb') as outf:
        #     pickle.dump(self.samples, outf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_image_idx, pos_images_idxs, neg_images_idxs = self.samples[idx]
        # todo: maybe add some scaling on all given descriptors
        gt_descriptor = self.train_gt_descriptors[gt_image_idx]
        pos_seq = self.train_df_descriptors[pos_images_idxs]
        neg_seq = self.train_df_descriptors[neg_images_idxs]

        gt_descriptor = torch.from_numpy(gt_descriptor)

        pos_seq = [torch.from_numpy(pos_img) for pos_img in pos_seq]
        neg_seq = [torch.from_numpy(neg_img) for neg_img in neg_seq]

        pos_seq = torch.stack(pos_seq, dim=0, out=None)
        neg_seq = torch.stack(neg_seq, dim=0, out=None)

        sample = {'gt_image': gt_descriptor,
                  'pos_seq': pos_seq,
                  'neg_seq': neg_seq}

        return sample


class FakeRNNMCSDataset(Dataset):

    def __init__(
            self,
            train_df: str,
            train_df_descriptors,
            train_gt_df,
            train_gt_descriptors,
            train_df_track_order_df,
            root_dir,
            is_val=False,
            transform=None
    ):
        seq_len = 5
        self.samples = [[np.random.randn(512).astype(np.float32),
                         [np.random.randn(512).astype(np.float32) for i in range(seq_len)],
                         [np.random.randn(512).astype(np.float32) for i in range(seq_len)]]
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

    dataset = RNNMCSDataset(
        train_df="../../data/raw/train_df.csv",
        train_df_descriptors="../../data/raw/train_df_descriptors.npy",
        train_gt_df="../../data/raw/train_gt_df.csv",
        train_gt_descriptors="../../data/raw/train_gt_descriptors.npy",
        train_df_track_order_df="../../data/raw/train_df_track_order_df.csv",
        root_dir='../../data/raw/data',
        is_val=False,
        transform=None
    )

    print(f"Total triples in {'test' if is_val else 'train'} dataset is {len(dataset)}")
    if iterate_data:
        for i in range(len(dataset)):
            sample = dataset[i]
            # print(sample['track_image'])

            print(i, sample['gt_image'].size(), sample['pos_seq'].size(), sample['neg_seq'].size())

            # if i == 3:
            #     break


if __name__ == '__main__':
    # example usage
    # python -i read_dataset.py check_data_iteration
    check_data_iteration(iterate_data=True)
