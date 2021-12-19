#!/bin/bash

source ~/.bashrc
module load python/3.7
module load pytorch/1.7
source $HOME/envs/oldl0/bin/activate

python $HOME/github/L0_regularization/train_wide_resnet.py --epochs 200 --dataset 'c10' --lamba 0.05 --multi_gpu 

# sbatch --output $HOME/slurm_logs/10_1.out --time=6:00:00 --gres=gpu:4 --mem=32G --cpus-per-task 10 $HOME/github/L0_regularization/script.sh