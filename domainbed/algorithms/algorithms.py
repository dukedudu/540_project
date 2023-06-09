# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted based on https://github.com/yaoxufeng/PCL-Proxy-based-Contrastive-Learning-for-Domain-Generalization
# and https://github.com/facebookresearch/DomainBed

import math
import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from timm.models.layers import trunc_normal_

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer


from domainbed.losses import ProxyLoss, ProxyPLoss, ClsCLLoss
torch.manual_seed(0)


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape

        print("==============input-shape==========", self.input_shape)

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape,
                                  num_classes, num_domains, hparams)
        self.encoder, self.scale, self.pcl_weights = networks.encoder(hparams)
        self._initialize_weights(self.encoder)
        self.fea_proj, self.fc_proj = networks.fea_proj(hparams)
        nn.init.kaiming_uniform_(self.fc_proj, mode='fan_out', a=math.sqrt(5))
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier = nn.Parameter(torch.FloatTensor(num_classes,
                                                         self.hparams['out_dim']))
        nn.init.kaiming_uniform_(
            self.classifier, mode='fan_out', a=math.sqrt(5))

        self.optimizer = torch.optim.Adam([
            {'params': self.featurizer.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.fea_proj.parameters()},
            {'params': self.fc_proj},
            {'params': self.classifier},
        ], lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

        self.proxycloss = ProxyPLoss(num_classes=num_classes, scale=self.scale)

    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        rep, pred = self.predict(all_x, style_aug=False)
        loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), all_y)

        fc_proj = F.linear(self.classifier, self.fc_proj)
        assert fc_proj.requires_grad == True
        loss = loss_cls
        if self.hparams['PCL_loss'] == 1:
            loss_pcl = self.proxycloss(rep, all_y, fc_proj)
            loss += self.pcl_weights * loss_pcl

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss_cls": loss_cls.item(), "loss_pcl": loss_pcl.item()}

    def predict(self, x, style_aug=False):
        self.featurizer.style_augment = style_aug
        self.featurizer.init = style_aug
        x = self.featurizer(x)
        x = self.encoder(x)
        rep = self.fea_proj(x)
        pred = F.linear(x, self.classifier)
        return rep, pred


class ACL(Algorithm):
    """
    Adversial Contrastive Loss
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ACL, self).__init__(input_shape,
                                  num_classes, num_domains, hparams)
        self.encoder, self.scale, self.pcl_weights = networks.encoder(hparams)
        self._initialize_weights(self.encoder)
        self.fea_proj, self.fc_proj = networks.fea_proj(hparams)
        nn.init.kaiming_uniform_(self.fc_proj, mode='fan_out', a=math.sqrt(5))
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier = nn.Parameter(torch.FloatTensor(num_classes,
                                                         self.hparams['out_dim']))  # changed
        self.domain_classifier = nn.Parameter(torch.FloatTensor(num_domains,
                                                                self.hparams['out_dim']))  # changed
        self.style_domain_classifier = nn.Parameter(torch.FloatTensor(num_domains,
                                                                      self.hparams['layer_stats_dim'][self.hparams['resnet18']][self.hparams['adversial_pos']]))  # changed
        nn.init.kaiming_uniform_(
            self.classifier, mode='fan_out', a=math.sqrt(5))

        self.adversial = self.hparams["adversial"]
        if self.adversial == 1:
            self.domainLabel = torch.LongTensor(
                [i for i in range(num_domains)]).reshape(num_domains, 1)
            self.domainLabel = torch.flatten(
                self.domainLabel.repeat((1, self.hparams['batch_size'])))
            nn.init.kaiming_uniform_(
                self.domain_classifier, mode='fan_out', a=math.sqrt(5))
            nn.init.kaiming_uniform_(
                self.style_domain_classifier, mode='fan_out', a=math.sqrt(5))
            self.optimizer_max = torch.optim.Adam([
                {'params': self.featurizer.parameters()},
                {'params': self.encoder.parameters()},
                {'params': self.fea_proj.parameters()},
            ], lr=self.hparams["lr"]/10, weight_decay=self.hparams["weight_decay"])

        self.optimizer = torch.optim.Adam([
            {'params': self.featurizer.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.fea_proj.parameters()},
            {'params': self.fc_proj},
            {'params': self.classifier},
            {'params': self.domain_classifier},
            {'params': self.style_domain_classifier},
        ], lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

        self.clsCLLoss = ClsCLLoss(num_classes=num_classes, scale=self.scale)
        self.cl_loss_weight = self.hparams["cl_loss_weight"]

    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        rep, pred, style_info = self.predict(all_x, style_adv=True)
        loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), all_y)

        fc_proj = F.linear(self.classifier, self.fc_proj)
        assert fc_proj.requires_grad == True

        loss = loss_cls
        if self.hparams["CLSCL_loss"] == 1:
            loss_triplets = self.clsCLLoss(rep, all_y, fc_proj)
            loss += self.cl_loss_weight*loss_triplets

        if self.adversial == 1:
            # loss = 0
            # rep, style_info = self.predict(all_x, style_adv=True)
            domain_pred = F.linear(rep, self.domain_classifier)
            domain_cls = F.nll_loss(F.log_softmax(domain_pred, dim=1),
                                    self.domainLabel.to(domain_pred.device))
            loss += domain_cls
            style_info_pred = F.linear(
                style_info, self.style_domain_classifier)
            style_domain_cls = F.nll_loss(F.log_softmax(
                style_info_pred, dim=1), self.domainLabel.to(domain_pred.device))
            loss += style_domain_cls
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.adversial == 1:
            rep, pred, style_info = self.predict(all_x, style_adv=True)
            cls_max = F.nll_loss(F.log_softmax(pred, dim=1), all_y)
            loss = cls_max
            domain_pred = F.linear(rep, self.domain_classifier)
            domain_cls_max = F.nll_loss(F.log_softmax(domain_pred, dim=1),
                                        self.domainLabel.to(domain_pred.device))
            loss -= domain_cls_max
            style_info_pred = F.linear(
                style_info, self.style_domain_classifier)
            style_domain_cls_max = F.nll_loss(F.log_softmax(
                style_info_pred, dim=1), self.domainLabel.to(domain_pred.device))
            loss -= style_domain_cls_max
            self.optimizer_max.zero_grad()
            loss.backward()
            self.optimizer_max.step()

        return {"loss_cls": loss_cls.item(),
                "loss_triplet": loss_triplets.item() if self.hparams['CLSCL_loss'] == 1 else 0,
                "loss_domain_cls": domain_cls.item() if self.adversial == 1 else 0,
                "loss_style_domain_cls": style_domain_cls.item() if self.adversial == 1 else 0,
                "cls_max": cls_max.item() if self.adversial == 1 else 0,
                "max_domain_cls": domain_cls_max.item() if self.adversial == 1 else 0,
                "max_style_domain_cls": style_domain_cls_max.item() if self.adversial == 1 else 0, }

    def predict(self, x, style_adv=False):
        x = self.featurizer(x, style_adv)
        x = self.encoder(x)
        rep = self.fea_proj(x)
        pred = F.linear(x, self.classifier)
        # if style_adv:
        #     return rep, self.featurizer.style_stats
        return rep, pred, self.featurizer.style_stats
