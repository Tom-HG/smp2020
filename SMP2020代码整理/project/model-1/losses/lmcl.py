#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:rossliang
# create time:2020/7/13 12:59 下午
import torch
from torch import nn
from torch.autograd import Variable


class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, logit_layer: nn.Linear, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = logit_layer.in_features
        self.num_classes = logit_layer.out_features
        self.s = s
        self.m = m
        self.logit_layer = logit_layer

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        centers = self.logit_layer.weight
        norms_c = torch.norm(centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(centers, norms_c)
        # logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1)) + self.logit_layer.bias

        y_onehot = torch.FloatTensor(batch_size, self.num_classes).to(self.logit_layer.weight.device)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


if __name__ == '__main__':
    logit_layer = nn.Linear(100, 10)
    lmcl = LMCL_loss(logit_layer)
    feat = torch.normal(0, 1, (3, 100))
    label = torch.randint(0, 9, (3,))
    print(lmcl(feat, label))
