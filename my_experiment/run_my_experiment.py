import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch.utils.data as Data
import nerual_network.my_mlp as mlp


def load_data():
    # 导入空车转移概率数据表
    path = './dataset/new_area_transition_probability'
    total_data = pd.read_table(path, header=None, engine='python', sep=',')
    x = total_data.iloc[:, 0:3:1].values  # numpy
    x_0 = total_data.iloc[:, 0:1:1].values
    x_12 = total_data.iloc[:, 1:3:1].values
    x_0 = normalize(x_0, axis=0, norm='max')  # 均值归一化
    # x_12 = normalize(x_12, axis=0, norm='max')  # 均值归一化
    x_12 = StandardScaler().fit_transform(x_12)  # 标准化

    # x_12 = -1 * np.log(x_12)

    x = np.c_[x_0, x_12]
    y = total_data.iloc[:, 3].values  # numpy

    print("原始数据{}异常".format("无" if not np.isnan(x).any() else "有"))
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    y = torch.unsqueeze(y, dim=1)
    kflod = KFold(n_splits=10)
    for train_index, test_index in kflod.split(x):
        # print("train_index: {} , test_index: {} ".format(train_index, test_index))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # return x_train, y_train, x_test, y_test
    dataset_train = Data.TensorDataset(x_train, y_train)
    dataset_test = Data.TensorDataset(x_test, y_test)

    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=2**10,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    loader_test = Data.DataLoader(
        dataset=dataset_test,
        batch_size=2**10,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    return loader_train, loader_test


if __name__ == '__main__':
    loader_train, loader_test = load_data()
    # x_train, y_train, x_test, y_test = load_data()

    if torch.cuda.is_available():
        my_mlp = mlp.my_mlp(loader_train, loader_test, epoch=2000,
                            input_size=3, hidden_size=3, output_size=1,
                            optimizer="SGD", loss_func="MSE", lr=1e-3)
        my_mlp.train_my_mlp()
        # net = mlp.train_mlp(loader_train, loader_test, 3, 3, 1, 500)
        # mlp.test_my_nn(net.to(use_gpu()), loader_test








