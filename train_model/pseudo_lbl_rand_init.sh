#!/bin/bash 

source activate <your_environment>

echo "no_of_tr_imgs, comb_tr_imgs: $1, $2"
echo "dataset" $3

echo start baseline training
python tr_baseline.py --no_of_tr_imgs=$1 --comb_tr_imgs=$2 --dataset=$3
echo end of baseline training

echo start joint training
python prop_method_joint_tr_rand_init.py --no_of_tr_imgs=$1 --comb_tr_imgs=$2 --dataset=$3 
echo end of joint training

