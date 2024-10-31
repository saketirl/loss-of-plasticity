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

python run_ppo.py -c cfg/ant_std_ls/ant_std_9_0.yml -s 9 -p 1

DIRECTORY="data/ant_std_ls/ant_std_9_25"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_9_0/9.pth "${DIRECTORY:?}"/25.pth

DIRECTORY="data/ant_std_ls/ant_std_9_26"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_9_0/9.pth "${DIRECTORY:?}"/26.pth

python run_ppo.py -c cfg/ant_std_ls/ant_std_9_25.yml -s 25 &
python run_ppo.py -c cfg/ant_std_ls/ant_std_9_26.yml -s 26

sleep 1200

#########################################################
# Execute the same block for a different seed
python run_ppo.py -c cfg/ant_std_ls/ant_std_10_0.yml -s 10 -p 1

DIRECTORY="data/ant_std_ls/ant_std_10_27"

# Check if the directory does not exist`
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_10_0/10.pth "${DIRECTORY:?}"/27.pth

DIRECTORY="data/ant_std_ls/ant_std_10_28"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/ant_std_10_0/10.pth "${DIRECTORY:?}"/28.pth

python run_ppo.py -c cfg/ant_std_ls/ant_std_10_27.yml -s 27 &
python run_ppo.py -c cfg/ant_std_ls/ant_std_10_28.yml -s 28

sleep 1200

python run_eval_ckpt.py -c cfg/ant/std.yml -m data/ant/ant_std_9_25/25.pth -f data/ant/ant_std_9_26/26.pth -o data/ant/ant_std_eval/eval_9_25_9_26.pkl &
python run_eval_ckpt.py -c cfg/ant/std.yml -m data/ant/ant_std_10_27/27.pth -f data/ant/ant_std_10_28/28.pth -o data/ant/ant_std_eval/eval_10_27_10_28.pkl
sleep 1200
