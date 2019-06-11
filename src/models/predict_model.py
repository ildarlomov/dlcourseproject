from src.models.triplet import TripletNet
from src.models.baseline_net import ResNetCaffe, BasicBlock
from src.data.read_dataset import InferenceMCSDataset
from src.data.baseline_transformers import ToTensor
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm

def get_descriptors(out_agg_path, dataset, model):
    inference_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    embeddings = []

    for batch in inference_loader:
        # just because костыль)))
        batch_inf = model(batch['track_image'], batch['track_image'], batch['track_image'])
        embeddings.append(batch_inf[2].numpy())

    embeddings = np.hstack(embeddings)
    np.save(out_agg_path, embeddings)




