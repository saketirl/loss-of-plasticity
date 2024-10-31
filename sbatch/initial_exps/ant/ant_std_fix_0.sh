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

python run_ppo.py -c cfg/ant/std_fix_0_1.yml -s 2 &
python run_ppo.py -c cfg/ant/std_fix_0_2.yml -s 3
sleep 1200

python run_eval_ckpt.py -c cfg/ant/std_fix_0.yml -m data/ant/std_fix_0_1/2.pth -f data/ant/std_fix_0_2/3.pth -o data/ant/std_fix_0_eval/std_fix_1_2_1_3.pkl
sleep 1200
