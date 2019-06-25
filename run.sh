#!/usr/bin/env bash
#python eval.py $1 $2 $3 $4
#wget https://my-fancy-net-weights.com/weights
export CUDA_VISIBLE_DEVICES=0

#our baseline
#wget -O best.pth https://www.dropbox.com/s/d1vk7uwjj4kfedv/best.pth\?dl\=0
#mcs baseline



#foo="Hello"
#foo="$foo World"
#echo $foo
branch="local"
#branch="leaderboard"

#python3 get_descriptors.py test_df.csv best.pth test_descriptors.npy data

pip install -r requirements.txt

if [[ "$branch" == "leaderboard" ]];

    then
        echo "Working with branch $branch"
        echo "In THEN"
        wget -O best.pth 'http://mcs2019-competition.visionlabs.ru/resnet_caffe_weights.pth'

        python3 descriptors_calculating.py \
                --root_path data \
                --df_path test_df.csv \
                --descriptors_path test_descriptors.npy \
                --weights_path best.pth

        python3 eval.py \
                --test_df_path test_df.csv \
                --track_order_df_path track_order_df.csv \
                --test_descriptors_path test_descriptors.npy \
                --agg_descriptors_path agg_descriptors.npy

    else
        echo "Working with branch $branch"
        echo "In ELSE"
#        wget -O best.pth 'http://mcs2019-competition.visionlabs.ru/resnet_caffe_weights.pth'

        python descriptors_calculating.py \
            --root_path data/raw/data \
            --df_path data/raw/train_df.csv \
            --descriptors_path models/baseline/logs/6/descriptors.npy \
            --weights_path models/baseline/logs/25/checkpoints/train.34.pth
#            --weights_path best.pth


        python eval.py \
            --test_df_path data/raw/train_df.csv \
            --track_order_df_path data/raw/train_df_track_order_df.csv \
            --test_descriptors_path models/baseline/logs/6/descriptors.npy \
            --agg_descriptors_path models/baseline/agg_descriptors.npy

	    python get_scores.py \
            --predicted_descr_path models/baseline/agg_descriptors.npy \
            --test_track_order_df data/raw/train_df_track_order_df.csv \
            --test_gt_df_path data/raw/train_gt_df.csv \
	        --gt_descriptors_path data/raw/train_gt_descriptors.npy \
	        --test_df_path data/raw/train_df.csv
fi


#python get_descriptors.py data/raw/train_df.csv models/baseline/logs/6/checkpoints/best.pth models/baseline/logs/6/descriptors.npy data/raw/data
#python eval.py data/raw/train_df.csv data/raw/train_df_track_order_df.csv models/baseline/logs/6/descriptors.npy models/baseline/agg_descriptors.npy

#default descriptors
#python eval.py data/raw/train_df.csv data/raw/train_df_track_order_df.csv data/raw/train_df_descriptors.npy models/baseline/agg_descriptors.npy