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

python run_ppo.py -c cfg/ant_std_ls/ant_std_5_0.yml -s 5 -p 1

DIRECTORY="data/ant_std_ls/ant_std_5_17"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_5_0/5.pth "${DIRECTORY:?}"/17.pth

DIRECTORY="data/ant_std_ls/ant_std_5_18"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_5_0/5.pth "${DIRECTORY:?}"/18.pth

python run_ppo.py -c cfg/ant_std_ls/ant_std_5_17.yml -s 17 &
python run_ppo.py -c cfg/ant_std_ls/ant_std_5_18.yml -s 18

sleep 1200

#########################################################
# Execute the same block for a different seed
python run_ppo.py -c cfg/ant_std_ls/ant_std_6_0.yml -s 6 -p 1

DIRECTORY="data/ant_std_ls/ant_std_6_19"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_6_0/6.pth "${DIRECTORY:?}"/19.pth

DIRECTORY="data/ant_std_ls/ant_std_6_20"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_6_0/6.pth "${DIRECTORY:?}"/20.pth

python run_ppo.py -c cfg/ant_std_ls/ant_std_6_19.yml -s 19 &
python run_ppo.py -c cfg/ant_std_ls/ant_std_6_20.yml -s 20

sleep 1200

python run_eval_ckpt.py -c cfg/ant/std.yml -m data/ant/ant_std_5_17/17.pth -f data/ant/ant_std_5_18/18.pth -o data/ant/ant_std_eval/eval_5_17_5_18.pkl &
python run_eval_ckpt.py -c cfg/ant/std.yml -m data/ant/ant_std_6_19/19.pth -f data/ant/ant_std_6_20/20.pth -o data/ant/ant_std_eval/eval_6_19_6_20.pkl
sleep 1200
