import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tqdm import tqdm


def main(conf):
    '''
    The code for calculating the ROC curve and mean distance
    '''
    agg_descr_arr = np.load(conf.predicted_descr_path)
    test_track_order_df = pd.read_csv(conf.test_track_order_df)
    test_gt_df = pd.read_csv(conf.test_gt_df_path)
    gt_descr = np.load(conf.gt_descriptors_path)

    dist_arr = []
    labels_arr = []

    for curr_person_id, subdf in tqdm(test_track_order_df.groupby('person_id')):
        gt_subdf = test_gt_df[test_gt_df.person_id == curr_person_id]
        for idx in subdf.index.values:
            for jdx in gt_subdf.index.values:
                dist = np.linalg.norm(agg_descr_arr[idx] - gt_descr[jdx])
                dist_arr.append(dist)
                labels_arr.append(1)

        gt_subdf = test_gt_df[test_gt_df.person_id != curr_person_id]
        for idx in subdf.index.values:
            for jdx in gt_subdf.index.values:
                dist = np.linalg.norm(agg_descr_arr[idx] - gt_descr[jdx])
                dist_arr.append(dist)
                labels_arr.append(0)

    dist_arr = np.array(dist_arr)
    labels_arr = np.array(labels_arr)

    mean_positive_dist = np.mean(dist_arr[labels_arr == 1])
    fpr, tpr, thr = roc_curve(labels_arr, -1 * dist_arr)

    tpr_filtered = tpr[fpr <= 1e-6]
    if len(tpr_filtered) == 0:
        needed_tpr = 0.0
    else:
        needed_tpr = tpr_filtered[-1]

    print('score 1 (tpr@fpr=1e-6): {0:.4f} score 2 (mean distance): {1:.4f}'.format(needed_tpr, mean_positive_dist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('predicted_descr_path', type=str, help='Path to predicted agg_descriptors.npy file')
    parser.add_argument('test_track_order_df', type=str, help='Path to test_track_order.csv file')
    parser.add_argument('test_gt_df_path', type=str, help='Path to test_gt_df.csv file')
    parser.add_argument('gt_descriptors_path', type=str, help='Path to gt_descriptors.npy file')
    conf = parser.parse_args()
    main(conf)
