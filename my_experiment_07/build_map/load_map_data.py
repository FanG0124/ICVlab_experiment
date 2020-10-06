'''
readCsvData
读取六边形网格数据，构建六边形网格地图
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 导入六边形网格数据
def load_map_data():
    map_path = '../dataset/new_transition_probability'
    data = pd.read_table(map_path, header=None)
    return data


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 5))

    x = load_map_data().loc[:, 1:4:2].values
    y = load_map_data().loc[:, 2:5:2].values
    # ax.scatter(x, y, s=10, c='b')
    ax.scatter(load_map_data().loc[1000, 1], load_map_data().loc[1000, 2], s=40, c='r')
    print(x)
    print(y)
    plt.show()
