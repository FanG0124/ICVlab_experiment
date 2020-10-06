'''
lstm网络
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class my_lstm(torch.nn.Module):
    '''
    input_size 为输入数据的大小
    cell_state_size 为细胞状态的大小
    hidden_state_size 为隐藏状态的大小
    '''
    def __init__(self, input_size, cell_state_size, hidden_state_size):
        super(my_lstm, self).__init__()

        self.cell_state_size = cell_state_size
        self.hidden_state_size = hidden_state_size
        # 忘记门限层
        self.forget_gate_layer = nn.Linear(input_size + hidden_state_size, hidden_state_size)
        # 输入门限层
        self.input_gate_layer = nn.Linear(input_size + hidden_state_size, hidden_state_size)
        # 输出门限层
        self.output_gate_layer = nn.Linear(input_size + hidden_state_size, hidden_state_size)
        # 新候选值层
        self.cand_cell_state_layer = nn.Linear(input_size + hidden_state_size, hidden_state_size)

    def step(self, input, cell_state, hidden_state):
        # 将输入与隐含状态拼接在一起
        combine_input_hidden = torch.cat((input, hidden_state), 1)
        # forget_t,input_t,input_t
        # 是一个 lstm 结构中三个门限
        # 经过 sigmoid 函数后 输出0-1的数值
        # 1为完全保留 , 0为完全放弃
        forget_t = F.sigmoid(self.forget_gate_layer(combine_input_hidden))
        input_t = F.sigmoid(self.input_gate_layer(combine_input_hidden))
        output_t = F.sigmoid(self.output_gate_layer(combine_input_hidden))
        # 新候选值
        cand_cell_state = F.tanh(self.cand_cell_state_layer(combine_input_hidden))
        # 将当前细胞状态cell state 和 隐含状态 hidden state 更新,向下一个lstm结果传递
        cell_state = forget_t * cell_state + input_t * cand_cell_state
        hidden_state = output_t * F.tanh(cell_state)

        return cell_state, hidden_state

    def forward(self, input):
        # batch_size为input的行数
        # time_step 为input的列数
        batch_size = input.size(0)
        time_step = input.size(1)
        cell_state, hidden_state = self.init_hidden(batch_size)
        output = None
        for i in range(time_step):
            cell_state, hidden_state = self.step(torch.squeeze(input[:, i:i+1, :]), cell_state, hidden_state)
            if output is None:
                output = hidden_state.unsqueeze(1)
            else:
                output = torch.cat((hidden_state.unsqueeze(1), output), 1)
        return output

    def init_hidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            cell_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return cell_state, hidden_state
        else:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size))
            cell_state = Variable(torch.zeros(batch_size, self.hidden_size))
            return cell_state, hidden_state




# torch.nn.LSTM


