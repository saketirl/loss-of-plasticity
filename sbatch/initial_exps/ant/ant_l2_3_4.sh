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

python run_ppo.py -c cfg/ant_l2_ls/l2_3_0.yml -s 3 -p 1

DIRECTORY="data/ant_l2_ls/l2_3_13"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/l2_3_0/3.pth "${DIRECTORY:?}"/13.pth

DIRECTORY="data/ant_l2_ls/l2_3_14"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/l2_3_0/3.pth "${DIRECTORY:?}"/14.pth

python run_ppo.py -c cfg/ant_l2_ls/l2_3_13.yml -s 13 &
python run_ppo.py -c cfg/ant_l2_ls/l2_3_14.yml -s 14

sleep 1200

#########################################################
# Execute the same block for a different seed
python run_ppo.py -c cfg/ant_l2_ls/l2_4_0.yml -s 4 -p 1

DIRECTORY="data/ant_l2_ls/l2_4_15"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/l2_4_0/4.pth "${DIRECTORY:?}"/15.pth

DIRECTORY="data/ant_l2_ls/l2_4_16"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/l2_4_0/4.pth "${DIRECTORY:?}"/16.pth

python run_ppo.py -c cfg/ant_l2_ls/l2_4_15.yml -s 15 &
python run_ppo.py -c cfg/ant_l2_ls/l2_4_16.yml -s 16

sleep 1200

python run_eval_ckpt.py -c cfg/ant/l2.yml -m data/ant/l2_3_13/13.pth -f data/ant/l2_3_14/14.pth -o data/ant/l2_eval/eval_3_13_3_14.pkl &
python run_eval_ckpt.py -c cfg/ant/l2.yml -m data/ant/l2_4_15/15.pth -f data/ant/l2_4_16/16.pth -o data/ant/l2_eval/eval_4_15_4_16.pkl
sleep 1200
