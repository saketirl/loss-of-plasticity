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

python run_ppo.py -c cfg/ant_std_ls/ant_std_11_0.yml -s 11 -p 1

DIRECTORY="data/ant_std_ls/ant_std_11_29"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_11_0/11.pth "${DIRECTORY:?}"/29.pth

DIRECTORY="data/ant_std_ls/ant_std_11_30"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_11_0/11.pth "${DIRECTORY:?}"/30.pth

python run_ppo.py -c cfg/ant_std_ls/ant_std_11_29.yml -s 29 &
python run_ppo.py -c cfg/ant_std_ls/ant_std_11_30.yml -s 30

sleep 1200

#########################################################
# Execute the same block for a different seed
python run_ppo.py -c cfg/ant_std_ls/ant_std_12_0.yml -s 12 -p 1

DIRECTORY="data/ant_std_ls/ant_std_12_31"

# Check if the directory does not exist`
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_12_0/12.pth "${DIRECTORY:?}"/31.pth

DIRECTORY="data/ant_std_ls/ant_std_12_32"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_12_0/12.pth "${DIRECTORY:?}"/32.pth

python run_ppo.py -c cfg/ant_std_ls/ant_std_12_31.yml -s 31 &
python run_ppo.py -c cfg/ant_std_ls/ant_std_12_32.yml -s 32

sleep 1200

python run_eval_ckpt.py -c cfg/ant/std.yml -m data/ant/ant_std_11_29/29.pth -f data/ant/ant_std_11_30/30.pth -o data/ant/ant_std_eval/eval_11_29_11_30.pkl &
python run_eval_ckpt.py -c cfg/ant/std.yml -m data/ant/ant_std_12_31/31.pth -f data/ant/ant_std_12_32/32.pth -o data/ant/ant_std_eval/eval_12_31_12_32.pkl
sleep 1200
