#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:rossliang
# create time:2020/7/14 9:32 下午
from torch import nn
from torch.nn import functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
