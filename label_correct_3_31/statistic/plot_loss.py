import pickle
import numpy as np
import matplotlib.pyplot as plt

iter_num = 8000
loss_file = "/home/dyt/Messi_du/2022/RDA/plot/TCL_loss_Office-31_" + str(iter_num) + "_plot.npz"

data = np.load(loss_file)
loss = data['arr_0']
noisy_label = data['arr_1']
real_label = data['arr_2']
loss = (loss - min(loss)) / (max(loss) - min(loss))

fig, ax = plt.subplots()
color = ['blue', 'red']
alpha = 0.7
ax.hist(loss[noisy_label == real_label], color=color[0], alpha=alpha, bins=np.arange(0, 1, 0.01))
ax.hist(loss[noisy_label != real_label], color=color[1], alpha=alpha, bins=np.arange(0, 1, 0.01))
# ax.hist(loss, color=color[1], alpha=alpha, bins=np.arange(0, 1, 0.01))
fig = plt.gcf()
plt.xlabel('normalized loss')
plt.ylabel('number')
# plt.legend(prop ={'size': 10})
fig.savefig('loss_office31' + str(iter_num) + '.png')

