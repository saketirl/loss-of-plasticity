from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import griddata

exp_dir_to_name = {
    'std': 'Standard PPO',
    'l2': 'PPO + L2 reg',
    'cbp': 'PPO + Continual Backprop'
}

if __name__ == '__main__':
    data_path = Path('/Users/ruoyutao/Documents/loss-of-plasticity/lop/rl/data/ant/std_eval/random_filters_3.pth')
    # data_path = Path('/Users/ruoyutao/Documents/loss-of-plasticity/lop/rl/data/ant/cbp_eval/random_filters_32.pth')
    # data_path = Path('/Users/ruoyutao/Documents/loss-of-plasticity/lop/rl/data/ant/l2_eval/random_filters_30.pth')

    name = data_path.parent.name.split('_')[0]

    f = open(data_path, 'rb')

    data_dict = pickle.load(f)

    tau_spaced = np.array(data_dict['returns'])[:, 0:2]
    returns_mean = np.array(data_dict['returns'])[:, 2]
    returns_std = np.array(data_dict['returns'])[:, 3]

    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]

    grid_return_mean = griddata(tau_spaced, returns_mean, (grid_x, grid_y), method='cubic')
    ret_min = -3000
    ret_max = 5000
    colorbar_title = 'returns'

    # grid_return_mean, ret_min, ret_max = np.log(grid_return_mean), np.log(ret_min), np.log(ret_max)
    # colorbar_title = 'log returns'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rax = axes[0]
    c1 = rax.contourf(grid_x, grid_y, grid_return_mean, levels=20, cmap="viridis",
                      vmin=ret_min, vmax=ret_max)
    cbar_1 = fig.colorbar(c1, ax=rax, label=colorbar_title)
    rax.legend()
    rax.set_title(f'Returns for {name}')

    # action jacobian norm mean and std different shape
    # tau_spaced = np.array(data_dict['action_jacobians'])[:, 0:2]
    # aj_mean = np.array(data_dict['action_jacobians'])[:, 2]
    # aj_std = np.array(data_dict['action_jacobians'])[:, 3]

    tau_spaced = []
    action_jacobian_means = []
    action_jacobian_stds = []
    action_jacobians_data = data_dict['action_jacobians']
    for i in range(len(action_jacobians_data)):
        tau_spaced.append(action_jacobians_data[i][0:2])
        action_jacobian_means.append(np.mean(action_jacobians_data[i][-2]))
        action_jacobian_stds.append(np.mean(action_jacobians_data[i][-1]))

    tau_spaced = np.array(tau_spaced)
    aj_mean = np.array(action_jacobian_means)
    aj_std = np.array(action_jacobian_stds)

    grid_act_jacob_mean = griddata(tau_spaced, aj_mean, (grid_x, grid_y), method='cubic')

    j_min, j_max = 0.275, 0.765

    ojax = axes[1]
    c2 = ojax.contourf(grid_x, grid_y, grid_act_jacob_mean , levels=20, cmap="viridis", vmin=j_min, vmax=j_max)
    cbar_2 = fig.colorbar(c2, ax=ojax, label='Action jacobian norm')
    ojax.legend()
    ojax.set_title(f'Action jacobian norm for {name}')

    fig.tight_layout()
    plt.show()

    fig_path = Path('/Users/ruoyutao/Documents/loss-of-plasticity/lop/rl/data/' + f'filter_norm_{name}.png')
    fig.savefig(fig_path)
    print(f"Saved figure to {fig_path}")


