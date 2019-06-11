from src.models.triplet import TripletNet
from src.models.baseline_net import ResNetCaffe, BasicBlock
from src.data.read_dataset import InferenceMCSDataset
from src.data.baseline_transformers import ToTensor
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
import torch

def get_descriptors(out_agg_path, dataset, model):
    inference_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    embeddings = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    c = 1
    for batch in tqdm(inference_loader):
        images = batch['track_image'].to(device)
        batch_inf = model.embeddingnet(images)
        embeddings.append(batch_inf.detach().cpu().numpy())
        c += 1
        # if c > 10: break

    embeddings = np.vstack(e for e in embeddings)
    print(f'Saved embeddings shape {embeddings.shape}')
    np.save(out_agg_path, embeddings)




