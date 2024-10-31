import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

data_file_path = sys.argv[1]
data_file_arr = data_file_path.split('/')
data_file_fname = data_file_arr[-2] + "_" + data_file_arr[-1].split('.')[0]

plt.rcParams.update({'font.size': 22})

def make_and_save_graph(spaced_tau, vals_mean, vals_std, ylabel, f_suffix):
    f_name = data_file_fname + f_suffix
    plt.figure(figsize=(10, 6))
    plt.plot(spaced_tau, vals_mean, color='b')
    plt.fill_between(spaced_tau, vals_mean - vals_std, vals_mean + vals_std, color='b', alpha=0.35)
    plt.xlabel(r'$\xi$')
    plt.ylabel(ylabel)
    plt.xlim(-0.35, 1.35)
    plt.title('Returns over Episodes')
    plt.axvline(x=0, color='r')
    plt.axvline(x=1, color='r')
    plt.grid(True)
    plt.savefig('plot_landscape/' + f_name + '.png')


with open(data_file_path, 'rb') as f:
    data_dict = pickle.load(f)

    tau_spaced = np.array(data_dict['returns'])[:, 0]
    returns_mean = np.array(data_dict['returns'])[:, 1]
    returns_std = np.array(data_dict['returns'])[:, 2]

    make_and_save_graph(tau_spaced, returns_mean, returns_std, 'Returns', data_file_fname + "_returns")

    tau_spaced = []
    action_jacobian_means = []
    action_jacobian_stds = []
    action_jacobians_data = data_dict['action_jacobians']
    for i in range(len(action_jacobians_data)):
        tau_spaced.append(action_jacobians_data[i][0])
        action_jacobian_means.append(np.mean(action_jacobians_data[i][-2]))
        action_jacobian_stds.append(np.mean(action_jacobians_data[i][-1]))

    tau_spaced = np.array(tau_spaced)
    action_jacobian_means = np.array(action_jacobian_means)
    action_jacobian_stds = np.array(action_jacobian_stds)

    make_and_save_graph(tau_spaced, action_jacobian_means, action_jacobian_stds, 'Output Jacobian',
                        data_file_fname + "_output")

    tau_spaced = np.array(data_dict['value_estimates'])[:, 0]
    values_jacobians_mean = np.array(data_dict['value_estimates'])[:, 1]
    values_jacobian_stds = np.array(data_dict['value_estimates'])[:, 2]
    print(values_jacobians_mean)
    print(values_jacobian_stds)
    make_and_save_graph(tau_spaced, values_jacobians_mean, values_jacobian_stds, 'Value Func Jacobian',
                        data_file_fname + "_value_fn")

    tau_spaced = np.array(data_dict['value_estimates'])[:, 0]
    values_estimates_mean = np.array(data_dict['value_estimates'])[:, 1]
    values_estimates_stds = np.array(data_dict['value_estimates'])[:, 2]

    make_and_save_graph(tau_spaced, values_estimates_mean, values_estimates_stds, 'Value Estimates',
                        data_file_fname + "_value_estimates")
