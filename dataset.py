import os
from PIL import Image

import torch.utils.data as data


def rgb_loader(path):
    return Image.open(path).convert('RGB')


class ImageListDataset(data.Dataset):
    """
    Builds a dataset based on a list of images.
    data_root - image path prefix
    data_list - image path array
    """

    def __init__(self, data_root, data_list, transform=None):
        self.data_root = data_root
        self.data_list = data_list
        self.transform = transform
        self.loader = rgb_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (input tensor) 
        """
        fpath = os.path.join(self.data_root, self.data_list[index])
        img = self.loader(fpath)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data_list)
