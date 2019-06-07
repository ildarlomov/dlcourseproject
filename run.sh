#!/usr/bin/env bash
#python eval.py $1 $2 $3 $4
#wget https://my-fancy-net-weights.com/weights
export CUDA_VISIBLE_DEVICES=0
python3 eval.py test_df.csv track_order_df.csv test_descriptors.npy agg_descriptors.npy
#python eval.py data/raw/train_df.csv data/raw/train_df_track_order_df.csv data/raw/train_df_descriptors.npy models/baseline/agg_descriptors.npy