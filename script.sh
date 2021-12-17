#!/bin/bash

source ~/.bashrc
module load python/3.7
module load pytorch/1.7
source $HOME/envs/oldl0/bin/activate

python $HOME/github/L0_regularization/train_wide_resnet.py --epochs 5 --dataset 'c10' --lamba 0.001 --multi_gpu 

# sbatch --output $HOME/slurm_logs/10_1.out --time=12:00:00 --gres=gpu:2 --mem=32G --cpus-per-task 10 $HOME/github/L0_regularization/script.sh