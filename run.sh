#!/usr/bin/env bash
#python eval.py $1 $2 $3 $4
#wget https://my-fancy-net-weights.com/weights
export CUDA_VISIBLE_DEVICES=0

wget -O best.pth https://www.dropbox.com/s/d1vk7uwjj4kfedv/best.pth\?dl\=0

python3 get_descriptors.py test_df.csv best.pth test_descriptors.npy data
python3 eval.py test_df.csv track_order_df.csv test_descriptors.npy agg_descriptors.npy

#python get_descriptors.py data/raw/train_df.csv models/baseline/logs/6/checkpoints/best.pth models/baseline/logs/6/descriptors.npy data/raw/data
#python eval.py data/raw/train_df.csv data/raw/train_df_track_order_df.csv models/baseline/logs/6/descriptors.npy models/baseline/agg_descriptors.npy

#default descriptors
#python eval.py data/raw/train_df.csv data/raw/train_df_track_order_df.csv data/raw/train_df_descriptors.npy models/baseline/agg_descriptors.npy