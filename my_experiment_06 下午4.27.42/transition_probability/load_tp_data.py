'''
导入空车转移概率数据
'''
import pandas as pd


def load_tp_data():
    # 声明文件路径
    path = '/Users/fangzhipeng/Desktop/idle_transition_probability'
    # 从文件中读取数据
    data = pd.read_table(path, header=None, engine='python', sep=','
                         , names=['Time', 'StartPos', 'EndPos', 'tran_probability'])
    # print(data.head())
    return data







