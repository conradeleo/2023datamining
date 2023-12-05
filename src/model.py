import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class GRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=1, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size= hidden_size#隐层大小
        self.num_layers = num_layers #gru层数
        # feature_size为特征维度,就是每个时间点对应的特征数量,这里为
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]# 获取批次大小
        #初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(1).float()
        else:
            h_0 = hidden

        # GRU运算
        # input matrix: [batch_size, sequence_length, feature_size]
        output, h_0 = self.gru(x, h_0)

        #获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape
        #将output变成 batch_size* timestep, hidden_dim
        output = output.reshape(-1, hidden_size)
        #全连接层
        output = self.fc(output)#形状为batch_size*timestep, 1
        #转换维度,用于输出
        output = output.reshape(batch_size, timestep, -1)

        #我们只需要返回最后一个时间片的数据即可
        return output
    
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-10
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred) + epsilon) / 2.0
        smape = torch.mean((numerator / denominator) * 100.0)
        return smape