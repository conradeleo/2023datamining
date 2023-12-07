import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class GRU(nn.Module):
    def __init__(self, input_size=5, feature_size=7, output_size=1, num_layers=1):
        super(GRU, self).__init__()
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, 1, num_layers, batch_first=True)
        self.fc = nn.Linear(feature_size, output_size)

    def forward(self, x, feature, hidden=None):
        batch_size = x.shape[0]

        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, 1).fill_(1).float()
        else:
            h_0 = hidden

        # input matrix: [batch_size, sequence_length, feature_size]
        output, h_0 = self.gru(x, h_0)

        batch_size, timestep, feature_size = output.shape
        output = output.reshape(batch_size, timestep, -1)
        #我们只需要返回最后一个时间片的数据即可

        combined_output = torch.cat([output, feature], dim=-1)
        output = self.fc(combined_output)
        return output
    
class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-10
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred) + epsilon) / 2.0
        smape = torch.mean((numerator / denominator) * 100.0)
        return smape