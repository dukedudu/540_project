# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted based on https://github.com/yaoxufeng/PCL-Proxy-based-Contrastive-Learning-for-Domain-Generalization
# and https://github.com/facebookresearch/DomainBed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

from domainbed.lib import wide_resnet
from domainbed.models.resnet import resnet50, resnet18

torch.manual_seed(0)


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                network = torchvision.models.resnet18(
                    pretrained=hparams["pretrained"])
                # network = resnet18(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 512
        else:
            if network is None:
                network = torchvision.models.resnet50(
                    pretrained=hparams["pretrained"])
                # network = resnet50(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:,
                                               i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        # del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

        # block x k x c
        self.style_dim = [256, 512, 1024]
        self.eps = 1e-6
        self.style_stats = None

    def forward(self, x, sty_adv=False):
        """Encode x into a feature vector of size n_outputs."""
        if not sty_adv:
            return self.dropout(self.network(x))

        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        x = self.network.layer1(x)
        if sty_adv and self.hparams['adversial_pos'] == 0:
            self.style_stats = self.compute_style(x)
        x = self.network.layer2(x)
        if sty_adv and self.hparams['adversial_pos'] == 1:
            self.style_stats = self.compute_style(x)
        x = self.network.layer3(x)
        if sty_adv and self.hparams['adversial_pos'] == 2:
            self.style_stats = self.compute_style(x)
        x = self.network.layer4(x)
        x = self.network.avgpool(x)
        x = self.network.fc(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x

    def update_style_bank(self, bank_stats, stats):
        # bank size: K x d
        # stats size: B x d
        # topk: select top k' dissimilar from stats
        similarity = F.normalize(stats) @ F.normalize(bank_stats).T  # B x K
        similarity = torch.softmax(similarity, dim=0)
        dissimilar_k = torch.topk(similarity, 1, largest=False, dim=0)
        selected_stats = stats[dissimilar_k.indices.T]  # K x K x d
        alpha = torch.sum(dissimilar_k.values.T, dim=1)
        new_stats = selected_stats * dissimilar_k.values.T.unsqueeze(2)
        bank_stats = (1 - alpha).unsqueeze(1) * \
            bank_stats + torch.sum(new_stats, dim=1)
        return bank_stats

    def compute_style(self, x):
        mu = x.mean(dim=[2, 3]).detach()
        var = x.var(dim=[2, 3]).detach()
        sig = (var + self.eps).sqrt()
        return torch.cat([F.normalize(mu), F.normalize(sig)], dim=1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(
            f"Input shape {input_shape} is not supported")


def encoder(hparams):
    if hparams["resnet18"] == False:
        n_outputs = 2048
    else:
        n_outputs = 512
    if hparams['dataset'] == "OfficeHome":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "PACS":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 256
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )

    elif hparams['dataset'] == "TerraIncognita":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    else:
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )

    return encoder, scale_weights, pcl_weights


def fea_proj(hparams):
    if hparams['dataset'] == "OfficeHome":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['hidden_size']),
            dropout,
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "PACS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 256
        hparams['out_dim'] = 256
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )

    elif hparams['dataset'] == "TerraIncognita":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    else:
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['hidden_size']),
            dropout,
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )

    return fea_proj, fc_proj
