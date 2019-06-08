import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        to_torch_tensor = lambda img: torch.from_numpy(img.transpose((2, 0, 1)))
        return {k: to_torch_tensor(v) for k, v in sample.items()}
