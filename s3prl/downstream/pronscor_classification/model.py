# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the 1-hidden model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""




import torch.nn.functional as F
import torch.nn as nn
import torch
from IPython import embed
class ConvBank(nn.Module):
    def __init__(self, input_dim, output_class_num, kernels, cnn_size, hidden_size, dropout, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout
        self.batch_norm = kwargs.get("batch_norm", False)

        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size,
                             kernel, padding=kernel//2))
        latest_size = cnn_size * len(kernels)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(latest_size)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        if self.batch_norm:
            hidden = self.bn(F.relu(hidden.transpose(1, 2))).transpose(1, 2)
        hidden = F.dropout(hidden, p=self.drop_p)

        predicted = self.out_linear(hidden)

        return predicted


class Linear(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Linear, self).__init__()

        # init attributes
        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        predicted = self.linear(features)
        return predicted
