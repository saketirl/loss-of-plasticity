#!/bin/bash

#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p 3090-gcondo
#SBATCH -t 36:00:00

module load anaconda/2023.09-0-7nso27y
module load cuda/11.8.0-lpttyok
module load mesa/22.1.6-6dbg5gq
source ~/.bash_profile
conda activate dynamicaldl

cd /users/stiwari4/data/stiwari4/dynamicaldl/loss-of-plasticity/lop/rl
rm data/ant/cbp_1/*
cp data/ant/cbp/1.pth data/ant/cbp_1/1.pth

python run_ppo.py -c cfg/ant/cbp_1.yml -s 1