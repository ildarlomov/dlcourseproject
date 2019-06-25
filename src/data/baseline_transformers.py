import torch


class TransformsWrapper(object):
    """Convert ndarrays in sample accoring to transforms."""

    def __init__(self, transform):
        self.transform = transform
        super().__init__()

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {k: self.transform(v) for k, v in sample.items()}
