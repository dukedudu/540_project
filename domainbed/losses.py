# coding: utf-8
# Adapted based on https://github.com/yaoxufeng/PCL-Proxy-based-Contrastive-Learning-for-Domain-Generalization
# and https://github.com/facebookresearch/DomainBed
'''
custom loss function
'''

import math
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

# Proposed losses

torch.manual_seed(0)


class ClsCLLoss(nn.Module):
    def __init__(self, num_classes, scale):
        super(ClsCLLoss, self).__init__()
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = scale
        self.criterion = torch.nn.TripletMarginLoss()

    def forward(self, feature, gt_label, proxy):
        feature = F.normalize(feature, p=2, dim=1)
        proxy = F.normalize(proxy, p=2, dim=1)
        pos_pred = proxy[gt_label]
        perm = torch.randperm(pos_pred.size(0)).to(feature.device)
        neg_pred = pos_pred[perm]
        loss = self.criterion(feature, pos_pred, neg_pred)
        return loss


# # =========================  proxy Contrastive loss (baseline) ==========================
class ProxyLoss(nn.Module):
    '''
    pass
    '''

    def __init__(self, scale=1, thres=0.1):
        super(ProxyLoss, self).__init__()
        self.scale = scale
        self.thres = thres

    def forward(self, feature, pred, target):
        feature = F.normalize(feature, p=2, dim=1)  # normalize
        feature = torch.matmul(feature, feature.transpose(1, 0))  # (B, B)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        feature = feature * ~label_matrix  # get negative matrix
        feature = feature.masked_fill(feature < self.thres, -np.inf)
        pred = torch.cat([pred, feature], dim=1)  # (N, C+N)

        loss = F.nll_loss(F.log_softmax(self.scale * pred, dim=1),
                          target)

        return loss


class ProxyPLoss(nn.Module):
    '''
    pass
    '''

    def __init__(self, num_classes, scale):
        super(ProxyPLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.LongTensor([i for i in range(num_classes)])
        self.scale = scale

    def forward(self, feature, target, proxy):
        feature = F.normalize(feature, p=2, dim=1)
        pred = F.linear(feature, F.normalize(
            proxy, p=2, dim=1))  # proxy class # (N, C)
        label = (self.label.unsqueeze(1).to(feature.device)
                 == target.unsqueeze(0))  # (C, N)
        pred = torch.masked_select(pred.transpose(1, 0), label)  # N,

        pred = pred.unsqueeze(1)  # (N, 1)

        feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)

        feature = feature * ~label_matrix  # get negative matrix
        feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)

        logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
        label = torch.zeros(logits.size(
            0), dtype=torch.long).to(feature.device)
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)

        return loss
