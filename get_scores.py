import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tqdm import tqdm


def report_scores(dist_arr, labels_arr):

    mean_positive_dist = np.mean(dist_arr[labels_arr == 1])
    fpr, tpr, thr = roc_curve(labels_arr, -1 * dist_arr)

    tpr_filtered = tpr[fpr <= 1e-6]
    if len(tpr_filtered) == 0:
        needed_tpr = 0.0
    else:
        needed_tpr = tpr_filtered[-1]

    print('score 1 (tpr@fpr=1e-6): {0:.4f} score 2 (mean distance): {1:.4f}'.format(needed_tpr, mean_positive_dist))


def main(conf):
    '''
    The code for calculating the ROC curve and mean distance
    '''
    agg_descr_arr = np.load(conf.predicted_descr_path)
    test_track_order_df = pd.read_csv(conf.test_track_order_df)
    test_gt_df = pd.read_csv(conf.test_gt_df_path)
    gt_descr = np.load(conf.gt_descriptors_path)
    # should include is_val column
    test_df = pd.read_csv(conf.test_df_path)

    is_val_df = test_df.groupby('person_id').agg({'is_val': lambda x: set(x).pop()}).reset_index(drop=False)
    test_track_order_df = pd.merge(test_track_order_df, is_val_df, on='person_id', how='left')
    order_df_train = test_track_order_df[test_track_order_df.is_val == False]
    order_df_test = test_track_order_df[test_track_order_df.is_val == True]

    def check_order_df(order_df):
        dist_arr = []
        labels_arr = []
        for curr_person_id, subdf in tqdm(order_df.groupby('person_id')):
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

        return dist_arr, labels_arr

    print('Train stage...')
    train_dists, train_labels = check_order_df(order_df_train)

    train_dists = np.array(train_dists)
    train_labels = np.array(train_labels)
    print(f'Train stage POS frac {np.round(np.mean(train_labels), 3)}')

    report_scores(train_dists, train_labels)

    print('Dev stage...')
    dev_dists, dev_labels = check_order_df(order_df_test)

    dev_dists = np.array(dev_dists)
    dev_labels = np.array(dev_labels)
    print(f'Dev stage POS frac {np.round(np.mean(dev_labels), 3)}')

    report_scores(dev_dists, dev_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference by list')
    parser.add_argument('--predicted_descr_path', type=str, help='Path to predicted agg_descriptors.npy file')
    parser.add_argument('--test_track_order_df', type=str, help='Path to test_track_order.csv file')
    parser.add_argument('--test_gt_df_path', type=str, help='Path to test_gt_df.csv file')
    parser.add_argument('--gt_descriptors_path', type=str, help='Path to gt_descriptors.npy file')
    parser.add_argument('--test_df_path', type=str, help='Path to test_df.csv file to get train/test split')
    conf = parser.parse_args()
    main(conf)
