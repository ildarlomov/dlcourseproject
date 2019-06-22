import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from src.data.rnn_dataset import FinalRNNMCSDataset
from src.models.triplet import RNNTripletNet
from src.models.rnn import ResnetBasedRNN
import os
import torchvision as tv
import torch
import torch.nn.functional as F


def main(conf):
    '''
    Baseline code.
    Let's just average all the track descriptors and normalize
    '''

    embedding_rnn = ResnetBasedRNN(
        embeddingnet=None,  # model, if you want to optimize baseline feature extractor as well
        num_layers=1,
        dropout=0,
        hidden_size=512,
        input_size=512
    )
    model = RNNTripletNet(embedding_rnn, pretrained=True, weights_path=conf.weights_path_lstm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD),
    ])

    # data
    test_ds = FinalRNNMCSDataset(
        test_df=conf.test_df_path,
        test_df_track_order_df=conf.track_order_df_path,
        test_descriptors_df=conf.test_descriptors_path,
        root_dir='data/raw/data',
        transform=preprocessing
    )


    dataset_loader = torch.utils.data.DataLoader(test_ds,
                                                 batch_size=256,
                                                 shuffle=False,
                                                 num_workers=6,
                                                 pin_memory=True,
                                                 drop_last=False)

    result_arr = np.empty((0, 512), dtype=np.float32)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
            data = data['img_seq'].to(device)
            output_y, (hn_y, cn_y) = model.embeddingnet(data)
            hn_y = torch.squeeze(hn_y)
            output = F.normalize(hn_y, dim=1)
            output = output.detach().cpu().numpy()
            result_arr = np.vstack((result_arr, output))

    np.save(conf.agg_descriptors_path, result_arr)

    # np.save(conf.agg_descriptors_path, agg_descr_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('--test_df_path', type=str, help='Path to test_df.csv')
    parser.add_argument('--track_order_df_path', type=str, help='Path to track_order_df.csv')
    parser.add_argument('--test_descriptors_path', type=str, help='Path to test_descriptors.npy')
    parser.add_argument('--agg_descriptors_path', type=str, help='Path to result train_df_agg_descriptors.npy')
    parser.add_argument('--weights_path_lstm', type=str, help='Path to my model for agg')

    conf = parser.parse_args()
    main(conf)
