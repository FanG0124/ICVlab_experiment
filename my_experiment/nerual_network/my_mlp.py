import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
from sklearn.metrics import explained_variance_score  # 可释方差也叫解释方差（explained_variance_score）
from sklearn.metrics import mean_absolute_error  # 平均绝对误差（mean_absolute_error）
from sklearn.metrics import mean_squared_error  # 均方误差（mean_squared_error）
from sklearn.metrics import median_absolute_error  # 中值绝对误差（median_absolute_error）
from sklearn.metrics import r2_score  # R方值，确定系数（r2_score）


# 定义mlp
class Mlp(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Mlp, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


class my_base_mlp():

    def __init__(self, input_size, hidden_size, output_size, optimizer, loss_func, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr = lr

    # 查看是否使用GPU
    def choose_device(self):
        ngpu = 1
        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        return device

    # 设置mlp参数
    def set_mlp_index(self):
        return Mlp(self.input_size, self.hidden_size, self.output_size).to(self.choose_device())

    # 设置优化器
    def set_mlp_optim(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.set_mlp_index().parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            return torch.optim.SGD(self.set_mlp_index().parameters(), lr=self.lr)

    # 设置loss函数
    def set_mlp_lossfunc(self):
        if self.loss_func == "MSE":
            return torch.nn.MSELoss()
        elif self.loss_func == "SmoothL1Loss":
            return torch.nn.SmoothL1Loss()
        elif self.loss_func == "RMSE":
            return mean_squared_error


class my_mlp(my_base_mlp):

    _converge_train = []
    _converge_test = []

    def __init__(self, train_set, test_set, epoch, input_size, hidden_size, output_size, optimizer, loss_func, lr):
        super(my_mlp, self).__init__(input_size, hidden_size, output_size, optimizer, loss_func, lr)
        self.train_set = train_set
        self.test_set = test_set
        self.epoch = epoch
        self.net = self.set_mlp_index()
        self.loss_function = self.set_mlp_lossfunc()

    # 训练训练集
    def train_train_set(self):
        for step_train, (x_train, y_train) in enumerate(self.train_set):
            # 将数据放入cuda
            x = Variable(x_train).to(self.choose_device())
            y = Variable(y_train).to(self.choose_device())

            # net = self.set_mlp_index()
            prediction_train = self.net(x)

            # lossfunc = self.set_mlp_lossfunc()
            loss_train = self.loss_function(prediction_train, y)

            self.set_mlp_optim().zero_grad()
            loss_train.backward()
            self.set_mlp_optim().step()
        print("    训练集loss:{}".format(loss_train))
        self._converge_train.append(loss_train)

    # 测试测试集
    def test_test_set(self):
        for step_test, (x_test, y_test) in enumerate(self.test_set):
            # 将数据放入cuda
            x = Variable(x_test).to(self.choose_device())
            y = Variable(y_test).to(self.choose_device())
            # net = self.set_mlp_index()
            prediction_test = self.net(x)
            # lossfunc = self.set_mlp_lossfunc()
            loss_test = self.loss_function(prediction_test, y)
        print("    测试集loss:{}".format(loss_test))
        self._converge_test.append(loss_test)

    # 导出数据
    def export_data(self):
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        path_train = "./dataset/converge_train"
        np.savetxt(path_train, self._converge_train, fmt='%.06f')
        path_test = "./dataset/converge_test"
        np.savetxt(path_test, self._converge_test, fmt='%.06f')

    # 将训练好的网络保存
    def export_nn_params(self):
        state = {'model': self.set_mlp_index().state_dict(),
                 'optimizer_adam': self.set_mlp_optim().state_dict(),
                 'epoch': self.epoch}
        torch.save(state, "./dataset/sd")
        print("网络保存完成")

    # 训练mlp
    def train_my_mlp(self):

        for t in range(self.epoch):
            start_time = datetime.datetime.now()
            self.train_train_set()

            self.test_test_set()
            end_time = datetime.datetime.now()
            print("第{}个epoch用时{}".format(t, end_time - start_time))
            self.export_data()
        self.export_nn_params()


# 勘察误差
def check_error(label, pre_y):
    label = label.cpu().detach().numpy()
    pre_y = pre_y.cpu().detach().numpy()
    print("可释方差(explained_variance_score)为: {}"
          .format(mean_absolute_error(label, pre_y)))
    print("平均绝对误差(mean_absolute_error)为: {}".format(mean_absolute_error(label, pre_y)))
    print("均方误差（mean_squared_error）为: {}".format(mean_squared_error(label, pre_y)))
    print("中值绝对误差（median_absolute_error）为: {}".format(median_absolute_error(label, pre_y)))
    print("R方值，确定系数（r2_score）为: {}".format(r2_score(label, pre_y)))


