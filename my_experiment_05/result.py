import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

epoch = 1709
# converge_train = []
path_train = 'dataset/converge_train'
path_test = 'dataset/converge_test'
data_train = pd.read_table(path_train, header=None, engine='python', sep=',')
data_test = pd.read_table(path_test, header=None, engine='python', sep=',')
data_train = data_train.values
# data_test = data_test.values

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(epoch), data_train, 'r', label='train', linewidth=0.6)  # np.arange()返回等差数组
ax.plot(np.arange(epoch), data_test, 'b', label='test', linewidth=0.6)
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(loc='upper center', bbox_to_anchor=(0.8, 1), ncol=2)
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title('Loss vs. Training Epoch')
plt.grid()
plt.show()

# print(torch.cuda.is_available())