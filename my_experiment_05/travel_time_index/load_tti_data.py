import pandas as pd


def load_tti_data():
    # 导入空车转移概率数据表
    path = '../dataset/boundary.txt'
    data = pd.read_table(path, header=None, engine='python', sep=',')
    print(data.head())
    print(data.shape)


if __name__ == '__main__':
    load_tti_data()
