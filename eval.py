import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

def main(conf):
    '''
    Baseline code.
    Let's just average all the track descriptors and normalize
    '''
    test_df = pd.read_csv(conf.test_df_path)
    test_descriptors = np.load(conf.test_descriptors_path)
    track_order_df = pd.read_csv(conf.track_order_df_path)

    agg_descr_arr = np.empty((len(track_order_df), 512), dtype=np.float32)
    for track_index, track_id in tqdm(enumerate(track_order_df.track_id.values),
                                      total=len(track_order_df)):
        needed_idxes = test_df[test_df.track_id == track_id].index.values
        curr_descr = test_descriptors[needed_idxes].mean(axis=0)
        curr_norm = np.linalg.norm(curr_descr)
        if curr_norm > 0:
            curr_descr = curr_descr / curr_norm
        agg_descr_arr[track_index] = curr_descr

    np.save(conf.agg_descriptors_path, agg_descr_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('test_df_path', type=str, help='Path to test_df.csv')
    parser.add_argument('track_order_df_path', type=str, help='Path to track_order_df.csv')
    parser.add_argument('test_descriptors_path', type=str, help='Path to test_descriptors.npy')
    parser.add_argument('agg_descriptors_path', type=str, help='Path to result train_df_agg_descriptors.npy')
    conf = parser.parse_args()
    main(conf)

