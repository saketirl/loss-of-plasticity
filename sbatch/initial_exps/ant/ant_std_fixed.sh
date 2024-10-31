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

python run_ppo.py -c cfg/ant/std_fixed.yml -s 1 -p 1

DIRECTORY="data/ant_std_fixed/std_1_2"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/std_fixed/1.pth "${DIRECTORY:?}"/2.pth

DIRECTORY="data/ant_std_fixed/std_1_3"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/std_fixed/1.pth "${DIRECTORY:?}"/3.pth

python run_ppo.py -c cfg/ant/std_fixed_1.yml -s 2 &
python run_ppo.py -c cfg/ant/std_fixed_2.yml -s 3
sleep 1200

python run_eval_ckpt.py -c cfg/ant/std_fixed.yml -m data/ant/std_fixed_1/2.pth -f data/ant/std_fixed_2/3.pth -o data/ant/std_fixed_eval/std_fixed_1_2_1_3.pkl
sleep 1200
