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


class StyleCLLoss(nn.Module):

    def __init__(self, scale):
        super(StyleCLLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.scale = scale
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.style_criterion = nn.TripletMarginLoss()

    def forward_contrastive_supervised(self, feature, feature_aug, pred, gt_label):
        # variant 1: (proposed) supervised contrastive learning
        # pos: self and its style augmented sample
        # neg: other samples from different classes in the batch
        feature = F.normalize(feature, p=2, dim=1)
        feature_aug = F.normalize(feature_aug, p=2, dim=1)
        feature_m = torch.matmul(feature, feature_aug.transpose(1, 0))
        label_matrix = gt_label.unsqueeze(1) == gt_label.unsqueeze(0)
        neg_pairs = feature_m * ~label_matrix
        neg_pairs = neg_pairs.masked_fill(neg_pairs < 1e-6, -np.inf)  # (N, N)

        A = torch.ones(feature.shape[0], 1, 1, dtype=torch.bool)
        pos_mask = torch.block_diag(*A)
        pos_pairs = feature_m[pos_mask.bool()].view(pos_mask.shape[0], -1)
        logits = torch.cat([pos_pairs, neg_pairs], dim=1)
        labels = torch.zeros(
            logits.shape[0], dtype=torch.long).to(feature.device)
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), labels)
        return loss

    def forward_contrastive_unsupervised(self, feature, feature_aug):
        # variant 2: (unsupervised contrastive loss: infoNCE)
        # pos -- self and its style augmented sample
        # neg -- all other images
        feature = torch.cat([feature, feature_aug], dim=0)
        feature_norm = F.normalize(feature, p=2, dim=1)
        feature_mat = torch.matmul(feature_norm, feature_norm.transpose(1, 0))
        labels = torch.cat([torch.arange(feature_aug.shape[0]).repeat_interleave(1) for i in range(2)], dim=0).to(
            feature.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        A = torch.ones(labels.shape[0], 1, 1, dtype=torch.bool)
        mask = torch.block_diag(*A)

        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = feature_mat[~mask].view(feature_mat.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(
            logits.shape[0], dtype=torch.long).to(feature.device)

        loss = self.criterion(logits, labels)
        return loss

    def forward_positive_only(self, feature, feature_aug, pred, gt_label):
        # variant 3: mse
        # pos -- self and its style augmented sample
        # neg -- no neg pairs
        feature = F.normalize(feature, p=2, dim=1)
        feature_aug = F.normalize(feature_aug, p=2, dim=1)
        loss = self.mse(feature, feature_aug)
        return loss

    def forward(self, feature, feature_aug, pred, gt_label):
        return self.forward_contrastive_supervised(feature, feature_aug, pred, gt_label)


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
        perm = torch.randperm(pos_pred.size(0))
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
