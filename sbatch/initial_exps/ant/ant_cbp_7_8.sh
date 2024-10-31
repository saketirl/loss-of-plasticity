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

python run_ppo.py -c cfg/ant_cbp_ls/cbp_7_0.yml -s 7 -p 1

DIRECTORY="data/ant_cbp_ls/cbp_7_21"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/cbp_7_0/7.pth "${DIRECTORY:?}"/21.pth

DIRECTORY="data/ant_cbp_ls/cbp_7_22"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/cbp_7_0/7.pth "${DIRECTORY:?}"/22.pth

python run_ppo.py -c cfg/ant_cbp_ls/cbp_7_21.yml -s 21 &
python run_ppo.py -c cfg/ant_cbp_ls/cbp_7_22.yml -s 22

sleep 1200

#########################################################
# Execute the same block for a different seed
python run_ppo.py -c cfg/ant_cbp_ls/cbp_8_0.yml -s 8 -p 1

DIRECTORY="data/ant_cbp_ls/cbp_8_23"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi


rm -rf "${DIRECTORY:?}"/*
cp data/ant/cbp_8_0/8.pth "${DIRECTORY:?}"/23.pth

DIRECTORY="data/ant_cbp_ls/cbp_8_24"

# Check if the directory does not exist
if [ ! -d "$DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$DIRECTORY"
  echo "Directory created: $DIRECTORY"
else
  echo "Directory already exists: $DIRECTORY"
fi

rm -rf "${DIRECTORY:?}"/*
cp data/ant/cbp_8_0/8.pth "${DIRECTORY:?}"/24.pth

python run_ppo.py -c cfg/ant_cbp_ls/cbp_8_23.yml -s 23 &
python run_ppo.py -c cfg/ant_cbp_ls/cbp_8_24.yml -s 24

sleep 1200


python run_eval_ckpt.py -c cfg/ant/cbp.yml -m data/ant/cbp_7_21/21.pth -f data/ant/cbp_7_22/22.pth -o data/ant/cbp_eval/eval_7_21_7_22.pkl &
python run_eval_ckpt.py -c cfg/ant/cbp.yml -m data/ant/cbp_8_23/23.pth -f data/ant/cbp_8_24/24.pth -o data/ant/cbp_eval/eval_8_23_8_24.pkl
sleep 1200