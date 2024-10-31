from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    data_path = Path('/home/taodav/Documents/loss-of-plasticity/lop/rl/data/ant/cbp_eval/random_filters_cbp.pth')
    f = open(data_path, 'rb')

    data_dict = pickle.load(f)
    tau_spaced = np.array(data_dict['returns'])[:, 0:2]
    returns_mean = np.array(data_dict['returns'])[:, 2]
    returns_std = np.array(data_dict['returns'])[:, 3]


    print()


