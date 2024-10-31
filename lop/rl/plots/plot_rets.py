import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys


plt.rcParams.update({'font.size': 22})

log_path = sys.argv[1]
log_path_arr = log_path.split('/')
log_path_fname = log_path_arr[-2] + "_" + log_path_arr[-1].split('.')[0]
window_size = 100

# Load the data from the pickle file
with open(log_path, 'rb') as f:
    data_dict = pickle.load(f)

# Extract 'rets' from the data dictionary
rets = data_dict['rets']


# Check if 'rets' is a NumPy array; if not, convert it
if not isinstance(rets, np.ndarray):
    rets = np.array(rets)

# Check if 'rets' has enough data points
if len(rets) >= window_size:
    # Compute the simple moving average
    running_average = np.convolve(rets, np.ones(window_size) / window_size, mode='valid')
else:
    print(f"Not enough data points to compute a running average with a window size of {window_size}.")
    running_average = np.array([])  # Empty array if not enough data

episodes = np.arange(window_size - 1, len(rets))

# Create a plot of the 'rets' array
plt.figure(figsize=(10, 6))
plt.plot(episodes, running_average)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Returns over Episodes')
plt.grid(True)
plt.savefig('plots_rets/' + log_path_fname + '.png')

