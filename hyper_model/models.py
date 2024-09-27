from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.autograd import Variable

import pdb
from efficientnet_pytorch import EfficientNet
from torchvision import models

from hyper_model.efficientnet_gn import EfficientNet_GN


class LocalOutput(nn.Module):
    '''
    output layer module
    '''

    def __init__(self, n_input=84, n_output=2, nonlinearity=False):
        super().__init__()
        self.nonlinearity = nonlinearity
        layers = []
        if nonlinearity:
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_input, n_output))
        layers.append(nn.Softmax(dim=1))
        self.layer = nn.Sequential(*layers)
        self.loss_CE = nn.CrossEntropyLoss()

    def forward(self, x, y):
        pred = self.layer(x)
        loss = self.loss_CE(pred, y)
        return pred, loss


class HyperSimpleNet(nn.Module):
    '''
    hypersimplenet for adult and synthetic experiments
    '''

    def __init__(self, args, device, user_used):
        super(HyperSimpleNet, self).__init__()
        self.n_users = args.num_users
        # usr_used = [i for i in range(self.n_users)]
        self.usr_used = user_used
        self.device = device
        self.dataset = args.dataset
        hidden_dim = args.hidden_dim
        self.hidden_dim = hidden_dim
        self.train_pt = args.train_pt
        self.input_ray = Variable(torch.FloatTensor([[1 / len(self.usr_used) for i in self.usr_used]])).to(device)
        self.init_ray(0)

        self.input_ray.requires_grad = True
        spec_norm = args.spec_norm
        layers = [
            spectral_norm(nn.Linear(len(self.usr_used), hidden_dim)) if spec_norm else nn.Linear(len(self.usr_used),
                                                                                                 hidden_dim)]
        for _ in range(args.n_hidden - 1):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self.input_dim = args.input_dim

        if self.dataset == "synthetic1" or self.dataset == "synthetic2" or args.dataset=='synthetic3':
            self.loss = nn.MSELoss(reduction='mean')
            self.output_dim = 1
            self.l1_weights = nn.Linear(hidden_dim, self.input_dim * 100)
            self.l1_bias = nn.Linear(hidden_dim, 1)
            self.l2_weights = nn.Linear(hidden_dim, self.output_dim * 100)
            self.l2_bias = nn.Linear(hidden_dim, 1)

    def forward(self, x, y, usr_id, input_ray=None):
        if input_ray != None:
            self.input_ray.data = input_ray
        feature = self.mlp(self.input_ray)
        l1_weight = self.l1_weights(feature).view(100, self.input_dim)
        l1_bias = self.l1_bias(feature).view(-1)
        l2_weight = self.l2_weights(feature).view(self.output_dim, 100)
        l2_bias = self.l2_bias(feature).view(-1)
        x = F.leaky_relu(F.linear(x, weight=l1_weight, bias=l1_bias), 0.2)
        x = F.linear(x, weight=l2_weight, bias=l2_bias)
        if self.dataset == "synthetic1" or self.dataset == "synthetic2" or self.dataset=='synthetic3':
            pred = x.flatten()

        y = y.float()
        loss = self.loss(pred, y)
        return pred, loss

    def init_ray(self, target_usr, f=1):
        if self.dataset == "synthetic1":
            big_usr = []
            big_usr_idx = []

            for usr in self.usr_used:
                if usr in [0, 1, 4, 5]:
                    big_usr.append(usr)
                    big_usr_idx.append(self.usr_used.index(usr))

            if self.train_pt == True and len(big_usr) != 0:
                if (len(self.input_ray.data.shape) == 1):
                    for i in range(len(self.usr_used)):
                        if i in big_usr_idx:
                            self.input_ray.data[i] = 1 / len(big_usr)
                        else:
                            self.input_ray.data[i] = 0

                elif (len(self.input_ray.data.shape) == 2):
                    for i in range(len(self.usr_used)):
                        if i in big_usr_idx:
                            self.input_ray.data[0, i] = 1 / len(big_usr)
                        else:
                            self.input_ray.data[0, i] = 0

            elif self.train_pt == True and len(big_usr) == 0:
                if (len(self.input_ray.data.shape) == 1):
                    for i in range(len(self.usr_used)):
                        self.input_ray.data[i] = 1 / len(self.usr_used)
                        # if i == target_usr:
                        #     self.input_ray.data[i] = 1
                        # else:
                        #     self.input_ray.data[i] = 0

                elif (len(self.input_ray.data.shape) == 2):
                    for i in range(len(self.usr_used)):
                        self.input_ray.data[0,i] = 1 / len(self.usr_used)
                        # if i == target_usr:
                        #     self.input_ray.data[0, i] = 1
                        # else:
                        #     self.input_ray.data[0, i] = 0
            else:
                self.input_ray.data.fill_(1 / len(self.usr_used))
        elif self.dataset == "synthetic2" or self.dataset == "synthetic3":
            self.input_ray.data.fill_(1 / len(self.usr_used))


class SimpleNet(nn.Module):

    def __init__(self, args, device):
        super().__init__()
        hidden_dim = args.hidden_dim
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 1)

        self.n_users = args.num_users
        usr_used = [i for i in range(self.n_users)]
        self.usr_used = usr_used
        self.device = device
        self.dataset = args.dataset

        if self.dataset == "synthetic1" or self.dataset == "synthetic2":
            self.loss = nn.MSELoss(reduction='mean')
            self.output_dim = 1

    def forward(self, x, y, usr_id, input_tay=None):

        x.view(-1, 20)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        if self.dataset == "synthetic1" or self.dataset == "synthetic2":
            pred = x.flatten()

        y = y.float()
        loss = self.loss(pred, y)
        return pred, loss


class Hypernet(nn.Module):
    '''
    Hypernet for CIFAR10 experiments
    '''

    def __init__(self, args, n_usrs, usr_used, device, n_classes=10, in_channels=3, n_kernels=16, hidden_dim=100,
                 spec_norm=False, n_hidden=2):
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        self.n_classes = n_classes
        self.n_users = n_usrs
        self.usr_used = usr_used
        self.device = device

        self.input_ray = Variable(torch.FloatTensor([[1 / len(usr_used) for i in usr_used]])).to(device)
        self.input_ray.requires_grad = True

        layers = [
            spectral_norm(nn.Linear(len(usr_used), hidden_dim)) if spec_norm else nn.Linear(len(usr_used), hidden_dim)]

        for _ in range(n_hidden):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        self.c1_weights = []
        self.c1_bias = []
        self.c2_weights = []
        self.c2_bias = []
        self.l1_weights = []
        self.l1_bias = []
        self.l2_weights = []
        self.l2_bias = []

        for _ in range(n_hidden - 1):
            self.c1_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c1_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_bias.append(nn.LeakyReLU(0.2, inplace=True))

        self.c1_weights = nn.Sequential(*(self.c1_weights + [
            spectral_norm(nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)) if spec_norm else nn.Linear(
                hidden_dim, self.n_kernels * self.in_channels * 5 * 5)]))
        self.c1_bias = nn.Sequential(*(self.c1_bias + [
            spectral_norm(nn.Linear(hidden_dim, self.n_kernels)) if spec_norm else nn.Linear(hidden_dim,
                                                                                             self.n_kernels)]))
        self.c2_weights = nn.Sequential(*(self.c2_weights + [spectral_norm(
            nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)) if spec_norm else nn.Linear(hidden_dim,
                                                                                                            2 * self.n_kernels * self.n_kernels * 5 * 5)]))
        self.c2_bias = nn.Sequential(*(self.c2_bias + [
            spectral_norm(nn.Linear(hidden_dim, 2 * self.n_kernels)) if spec_norm else nn.Linear(hidden_dim,
                                                                                                 2 * self.n_kernels)]))
        self.l1_weights = nn.Sequential(*(self.l1_weights + [
            spectral_norm(nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)) if spec_norm else nn.Linear(
                hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)]))
        self.l1_bias = nn.Sequential(
            *(self.l1_bias + [spectral_norm(nn.Linear(hidden_dim, 120)) if spec_norm else nn.Linear(hidden_dim, 120)]))
        self.l2_weights = nn.Sequential(*(self.l2_weights + [
            spectral_norm(nn.Linear(hidden_dim, 84 * 120)) if spec_norm else nn.Linear(hidden_dim, 84 * 120)]))
        self.l2_bias = nn.Sequential(
            *(self.l2_bias + [spectral_norm(nn.Linear(hidden_dim, 84)) if spec_norm else nn.Linear(hidden_dim, 84)]))

        self.locals = nn.ModuleList([LocalOutput(n_output=n_classes) for i in range(self.n_users)])

    def forward(self, x, y, usr_id, input_ray=None):
        if input_ray != None:
            self.input_ray.data = input_ray.to(self.device)

        feature = self.mlp(self.input_ray)

        weights = {
            "conv1.weight": self.c1_weights(feature).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(feature).view(-1),
            "conv2.weight": self.c2_weights(feature).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(feature).view(-1),
            "fc1.weight": self.l1_weights(feature).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(feature).view(-1),
            "fc2.weight": self.l2_weights(feature).view(84, 120),
            "fc2.bias": self.l2_bias(feature).view(-1),
        }
        x = F.conv2d(x, weight=weights['conv1.weight'], bias=weights['conv1.bias'], stride=1)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, weight=weights['conv2.weight'], bias=weights['conv2.bias'], stride=1)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(F.linear(x, weight=weights["fc1.weight"], bias=weights["fc1.bias"]), 0.2)
        logits = F.leaky_relu(F.linear(x, weight=weights["fc2.weight"], bias=weights["fc2.bias"]), 0.2)
        index = self.usr_used.index(usr_id)

        pred, loss = self.locals[index](logits, y)

        return pred, loss

    def init_ray(self, target_usr, f=1):

        if self.args.train_pt == True:
            self.input_ray.data.fill_(1 / len(self.usr_used))
        else:

            if (len(self.input_ray.data.shape) == 1):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[i] = 0.8
                        else:
                            self.input_ray.data[i] = (1.0 - 0.8) / (len(self.usr_used) - 1)
            elif (len(self.input_ray.data.shape) == 2):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0, 0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[0, i] = 0.8
                        else:
                            self.input_ray.data[0, i] = (1.0 - 0.8) / (len(self.usr_used) - 1)

class CNNHyper100(nn.Module):
    def __init__(
            self,args,usr_used, n_nodes,device,in_channels=3, out_dim=100, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=2):
        super().__init__()

        self.args = args
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        self.n_classes = out_dim
        self.out_dim = out_dim
        self.n_users = n_nodes
        self.usr_used = usr_used
        self.device = device

        self.input_ray = Variable(torch.FloatTensor([[1 / len(usr_used) for i in usr_used]])).to(device)
        self.input_ray.requires_grad = True


        layers = [
            spectral_norm(nn.Linear(len(usr_used), hidden_dim)) if spec_norm else nn.Linear(len(usr_used), hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()

        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self,x,y,usr_id,input_ray = None):
        if input_ray!=None:
            self.input_ray.data = input_ray.to(self.device)
        features = self.mlp(self.input_ray)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        x = F.max_pool2d(F.relu(F.conv2d(x, weight=weights['conv1.weight'], bias=weights['conv1.bias'])),(2,2))
        x = F.max_pool2d(F.relu(F.conv2d(x, weight=weights['conv2.weight'], bias=weights['conv2.bias'])),(2,2))
        x = x.view(x.shape[0], -1)
        x = F.relu(F.linear(x,weight=weights["fc1.weight"],bias=weights["fc1.bias"]))
        x = F.relu(F.linear(x,weight=weights["fc2.weight"],bias=weights["fc2.bias"]))
        x = F.linear(x,weight=weights["fc3.weight"],bias=weights["fc3.bias"])
        loss = self.loss(x,y)
        # return weights
        return x,loss

    def init_ray(self, target_usr, f=1):

        if self.args.train_pt == True:
            self.input_ray.data.fill_(1 / len(self.usr_used))
        else:

            if (len(self.input_ray.data.shape) == 1):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[i] = 0.8
                        else:
                            self.input_ray.data[i] = (1.0 - 0.8) / (len(self.usr_used) - 1)
            elif (len(self.input_ray.data.shape) == 2):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0, 0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if i == target_usr:
                            self.input_ray.data[0, i] = 0.8
                        else:
                            self.input_ray.data[0, i] = (1.0 - 0.8) / (len(self.usr_used) - 1)


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class Basenet_cifar(nn.Module):
    def __init__(self, n_usrs, usr_used, device, n_classes=10, in_channels=3, n_kernels=16):
        super().__init__()
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        self.n_classes = n_classes
        self.n_users = n_usrs
        self.usr_used = usr_used
        self.device = device

        self.conv1 = nn.Conv2d(self.in_channels, self.n_kernels, 5)
        self.conv2 = nn.Conv2d(self.n_kernels, 2 * self.n_kernels, 5)
        self.fc1 = nn.Linear(2 * self.n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y, usr_id):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        pred = self.softmax(self.fc3(x))
        loss = self.loss(pred, y)
        return pred, loss


import torch, torch.optim, torch.nn as nn, torch.nn.functional as F
from torch.nn import TransformerEncoder
from collections import OrderedDict
import math

#################################################################################

task_dims = {
    'disch_24h': 10,
    'disch_48h': 10,
    'Final Acuity Outcome': 12,
    'tasks_binary_multilabel': 3,  # Mort24, Mort48, Long Length of stay
    'next_timepoint': 15,
    'next_timepoint_was_measured': 15,
    'masked_imputation': 15 * 2,
    'mort_24h': 1,
    'mort_48h': 1,
    'LOS': 1
}


########################################################################################################

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel1(nn.Module):

    def __init__(self, task='mort_48h', norm_layer='gn', ntoken=128, d_model=128, nhead=8, d_hid=128,
                 fc_layer_sizes=[128, 256],device=None,
                 nlayers=2, dropout=0.5):  # d_model = 128 , d_hid=384
        super(TransformerModel1, self).__init__()

        self.task = task
        self.device = device


        self.ts_continuous_projector = nn.Linear(165, d_model)  # 16
        self.statics_continuous_projector = nn.Linear(15, d_model)  # 16

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        if norm_layer == "gn":
            encoder_layers = TransformerEncoderLayer(d_model, nhead, norm_layer="gn", dim_feedforward=d_hid,
                                                     dropout=dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model, nhead, norm_layer="ln", dim_feedforward=d_hid,
                                                     dropout=dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        output_dim = d_model
        # self.norm = nn.BatchNorm1d(output_dim)

        fc_stack = OrderedDict()
        for i, fc_layer_size in enumerate(fc_layer_sizes):
            fc_stack[f"fc_{i}"] = nn.Linear(output_dim, fc_layer_size)

            if norm_layer == "gn":
                fc_stack[f"norm_{i}"] = nn.GroupNorm(2, fc_layer_size, eps=1e-5)
            elif norm_layer == "ln":
                fc_stack[f"norm_{i}"] = nn.LayerNorm(fc_layer_size, eps=1e-5)
            elif norm_layer == "bn":
                fc_stack[f"norm_{i}"] = nn.BatchNorm1d(fc_layer_size, eps=1e-5)

            fc_stack[f"relu_{i}"] = nn.ReLU()
            output_dim = fc_layer_size

        fc_stack["fc_last"] = nn.Linear(output_dim, 256)

        self.fc_stack = nn.Sequential(fc_stack)

        self.task_dim = task_dims[task]
        self.task_layer = nn.Linear(256, self.task_dim)

    def forward(self, X):
        ts_continuous = X[0].to(self.device).float()
        statics = X[1].to(self.device).float()
        input_sequence = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_continuous_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous

        output = self.pos_encoder(input_sequence)
        output = self.transformer_encoder(output)

        output = output.mean(dim=1)  # avg

        # output = self.bn(output)
        output = self.fc_stack(output)
        output = self.task_layer(output)

        if self.task in ['tasks_binary_multilabel', 'mort_24h', 'mort_48h', 'LOS']:
            output = nn.Sigmoid()(output)

        return output

class TransformerHyper(nn.Module):

    def __init__(self, args,n_nodes,embedding_dim,task='mort_48h', norm_layer='ln', ntoken=128, d_model=128, nhead=8, d_hid=128,
                 fc_layer_sizes=[128, 256],
                 nlayers=2, dropout=0.5, user_used=None, device=None):  # d_model = 128 , d_hid=384
        super(TransformerHyper, self).__init__()

        self.task = task
        self.usr_used = user_used
        self.ts_continuous_projector = nn.Linear(165, d_model)
        self.statics_continuous_projector = nn.Linear(15, d_model)
        hidden_dim = args.hidden_dim
        self.device = device
        self.dataset = args.dataset
        self.n_users = args.num_users
        self.train_pt = args.train_pt
        self.dropout = nn.Dropout(0.05)
        self.baseline_type = args.baseline_type
        self.fc_layer_sizes = fc_layer_sizes
        self.input_ray = Variable(torch.FloatTensor([[1 / len(self.usr_used) for i in self.usr_used]])).to(device)
        self.input_ray.requires_grad = True
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        self.norm_layer = norm_layer
        spec_norm = args.spec_norm
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(args.n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )
        self.mlp = nn.Sequential(*layers)

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)


        encoder_layers = TransformerEncoderLayer(d_model, nhead, norm_layer="ln", dim_feedforward=d_hid,
                                                     dropout=dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        output_dim = d_model
        # self.norm = nn.BatchNorm1d(output_dim)

        fc_stack = OrderedDict()
        for i, fc_layer_size in enumerate(fc_layer_sizes):
            fc_stack[f"fc_{i}"] = nn.Linear(output_dim, fc_layer_size)


            fc_stack[f"norm_{i}"] = nn.LayerNorm(fc_layer_size, eps=1e-5)


            fc_stack[f"relu_{i}"] = nn.ReLU()
            output_dim = fc_layer_size

        fc_stack["fc_last"] = nn.Linear(output_dim, 256)

        self.fc_stack = nn.Sequential(fc_stack)

        self.task_dim = task_dims[task]
        self.task_layer = nn.Linear(256, self.task_dim)
        self.l1_weights = nn.Linear(hidden_dim, 256 * self.task_dim)
        self.l1_bias = nn.Linear(hidden_dim, self.task_dim)
        self.l2_weights = nn.Linear(hidden_dim, 165 * d_model)
        self.l2_bias = nn.Linear(hidden_dim, d_model)
        self.l3_weights = nn.Linear(hidden_dim, 15 * d_model)
        self.l3_bias = nn.Linear(hidden_dim, d_model)
        self.l4_weights = nn.Linear(hidden_dim, d_model * 128)
        self.l4_bias = nn.Linear(hidden_dim, 128)
        self.l5_weights = nn.Linear(hidden_dim, 256 * 128)
        self.l5_bias = nn.Linear(hidden_dim, 256)
        self.l6_weights = nn.Linear(hidden_dim, 256 * 256)
        self.l6_bias = nn.Linear(hidden_dim, 256)
        self.loss =  nn.BCELoss()

    def forward(self,idx):
        emd = self.embeddings(idx)
        feature = self.mlp(emd)
        weights = OrderedDict({
            "ts_continuous_projector.weight" :self.dropout(self.l2_weights(feature).view(self.d_model, -1)),
            "ts_continuous_projector.bias": self.dropout(self.l2_bias(feature).view(self.d_model)),
            "statics_continuous_projector.weight":self.dropout(self.l3_weights(feature).view(self.d_model, -1)),
            "statics_continuous_projector.bias":self.dropout( self.l3_bias(feature).view(self.d_model)),
            "fc0.weight":self.dropout( self.l4_weights(feature).view(128, self.d_model) ),
            "fc0.bias":self.dropout( self.l4_bias(feature).view(128)),
            "fc1.weight":self.dropout(self.l5_weights(feature).view(256, 128)),
            "fc1.bias":self.dropout(self.l5_bias(feature).view(256) ),
            "layer.weight":self.dropout( self.l6_weights(feature).view(256, 256)),
            "layer.bias":self.dropout(self.l6_bias(feature).view(256)),
            "task_layer.weight": self.dropout(self.l1_weights(feature).view(self.task_dim, -1)),
            "task_layer.bias": self.dropout(self.l1_bias(feature).view(-1)),
        })

        return weights

class TransformerTarget(nn.Module):

    def __init__(self, args, task='mort_48h', norm_layer='ln', ntoken=128, d_model=128, nhead=8, d_hid=128,
                 fc_layer_sizes=[128, 256],
                 nlayers=2, dropout=0.5, user_used=None, device=None):  # d_model = 128 , d_hid=384
        super(TransformerTarget, self).__init__()

        self.task = task
        self.usr_used = user_used
        self.ts_continuous_projector = nn.Linear(165, d_model)
        self.statics_continuous_projector = nn.Linear(15, d_model)
        hidden_dim = args.hidden_dim
        self.device = device
        self.d_hid = d_hid
        self.dataset = args.dataset
        self.nhead = nhead
        self.n_users = args.num_users
        self.train_pt = args.train_pt
        self.dropout = nn.Dropout(0.05)
        self.baseline_type = args.baseline_type
        self.fc_layer_sizes = fc_layer_sizes
        self.nlayers = nlayers
        self.input_ray = Variable(torch.FloatTensor([[1 / len(self.usr_used) for i in self.usr_used]])).to(device)
        self.input_ray.requires_grad = True
        self.norm_layer = norm_layer
        spec_norm = args.spec_norm


        self.d_model = d_model


        self.ts_continuous_projector = nn.Linear(165, d_model)  # 16
        self.statics_continuous_projector = nn.Linear(15, d_model)  # 16



        output_dim = d_model
        # self.norm = nn.BatchNorm1d(output_dim)

        for i, fc_layer_size in enumerate(fc_layer_sizes):
            if i == 0:
                self.fc0 = nn.Linear(output_dim, fc_layer_size)
            else:
                self.fc1 = nn.Linear(output_dim, fc_layer_size)
            output_dim = fc_layer_size

        self.layer = nn.Linear(output_dim, 256)



        self.task_dim = task_dims[task]
        self.task_layer = nn.Linear(256, self.task_dim)

        self.loss =  nn.BCELoss()

    def forward(self, ts_continuous, statics):
        input_sequence = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_continuous_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous
        pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        output = pos_encoder(input_sequence)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.nhead, norm_layer="ln", dim_feedforward=self.d_hid,
                                                 dropout=self.dropout)
        transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        output = transformer_encoder(output)

        output = output.mean(dim=1)  # avg
        for i, fc_layer_size in enumerate(self.fc_layer_sizes):
            if i == 0:
                output = self.fc0(output)
            else:
                output = self.fc1(output)


            output = F.layer_norm(output, normalized_shape=[self.fc_layer_sizes[i]], eps=1e-5)


            F.relu(output)

        output = self.layer(output)

        output = self.task_layer(output)

        if self.task in ['tasks_binary_multilabel', 'mort_24h', 'mort_48h', 'LOS']:
            output = nn.Sigmoid()(output)



        return output




class TransformerModel(nn.Module):

    def __init__(self, args, task='mort_48h', norm_layer='ln', ntoken=128, d_model=128, nhead=8, d_hid=128,
                 fc_layer_sizes=[128, 256],
                 nlayers=2, dropout=0.5, user_used=None, device=None):  # d_model = 128 , d_hid=384
        super(TransformerModel, self).__init__()

        self.task = task
        self.usr_used = user_used
        self.ts_continuous_projector = nn.Linear(165, d_model)
        self.statics_continuous_projector = nn.Linear(15, d_model)
        hidden_dim = args.hidden_dim
        self.device = device
        self.dataset = args.dataset
        self.n_users = args.num_users
        self.train_pt = args.train_pt
        self.dropout = nn.Dropout(0.05)
        self.baseline_type = args.baseline_type
        self.fc_layer_sizes = fc_layer_sizes
        self.input_ray = Variable(torch.FloatTensor([[1 / len(self.usr_used) for i in self.usr_used]])).to(device)
        self.input_ray.requires_grad = True
        self.norm_layer = norm_layer
        spec_norm = args.spec_norm
        layers = [
            spectral_norm(nn.Linear(len(self.usr_used), hidden_dim)) if spec_norm else nn.Linear(len(self.usr_used),
                                                                                                 hidden_dim)]
        for _ in range(args.n_hidden - 1):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)


        encoder_layers = TransformerEncoderLayer(d_model, nhead, norm_layer="ln", dim_feedforward=d_hid,
                                                     dropout=dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        output_dim = d_model
        # self.norm = nn.BatchNorm1d(output_dim)

        fc_stack = OrderedDict()
        for i, fc_layer_size in enumerate(fc_layer_sizes):
            fc_stack[f"fc_{i}"] = nn.Linear(output_dim, fc_layer_size)


            fc_stack[f"norm_{i}"] = nn.LayerNorm(fc_layer_size, eps=1e-5)


            fc_stack[f"relu_{i}"] = nn.ReLU()
            output_dim = fc_layer_size

        fc_stack["fc_last"] = nn.Linear(output_dim, 256)

        self.fc_stack = nn.Sequential(fc_stack)

        self.task_dim = task_dims[task]
        self.task_layer = nn.Linear(256, self.task_dim)
        self.l1_weights = nn.Linear(hidden_dim, 256 * self.task_dim)
        self.l1_bias = nn.Linear(hidden_dim, self.task_dim)
        self.l2_weights = nn.Linear(hidden_dim, 165 * d_model)
        self.l2_bias = nn.Linear(hidden_dim, d_model)
        self.l3_weights = nn.Linear(hidden_dim, 15 * d_model)
        self.l3_bias = nn.Linear(hidden_dim, d_model)
        self.l4_weights = nn.Linear(hidden_dim, d_model * 128)
        self.l4_bias = nn.Linear(hidden_dim, 128)
        self.l5_weights = nn.Linear(hidden_dim, 256 * 128)
        self.l5_bias = nn.Linear(hidden_dim, 256)
        self.l6_weights = nn.Linear(hidden_dim, 256 * 256)
        self.l6_bias = nn.Linear(hidden_dim, 256)
        self.loss =  nn.BCELoss()

    def forward(self, ts_continuous, statics, input_ray=None):
        # ts_continuous = X[0].to(self.device).float()
        # statics = X[1].to(self.device).float()
        # y.to(self.device)

        if input_ray != None:
            self.input_ray.data = input_ray

        feature = self.mlp(self.input_ray)
        # l2_weight = self.dropout(self.l2_weights(feature).view(self.d_model,-1))
        # l2_bias = self.dropout(self.l2_bias(feature).view(-1))
        # self.ts_continuous_projector.weight = nn.Parameter(l2_weight)
        # self.ts_continuous_projector.bias = nn.Parameter(l2_bias)
        # l3_weight = self.dropout(self.l3_weights(feature).view(self.d_model, -1))
        # l3_bias = self.dropout(self.l3_bias(feature).view(-1))
        #
        # self.statics_continuous_projector.weight = nn.Parameter(l3_weight)
        # self.statics_continuous_projector.bias = nn.Parameter(l3_bias)
        # input_sequence = self.ts_continuous_projector(ts_continuous)
        # statics_continuous = self.statics_continuous_projector(statics)
        l2_weight =self.dropout(self.l2_weights(feature).view(self.d_model, -1))
        l2_bias = self.dropout(self.l2_bias(feature).view(self.d_model))
        l3_weight = self.dropout(self.l3_weights(feature).view(self.d_model, -1))
        l3_bias =self.dropout( self.l3_bias(feature).view(self.d_model))

        input_sequence = F.linear(ts_continuous, l2_weight, l2_bias)
        statics_continuous = F.linear(statics, l3_weight, l3_bias)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous

        output = self.pos_encoder(input_sequence)
        output = self.transformer_encoder(output)

        output = output.mean(dim=1)  # avg
        l4_weight =self.dropout( self.l4_weights(feature).view(128, self.d_model) ) # Reshape to [128, d_model]
        l4_bias =self.dropout( self.l4_bias(feature).view(128))  # Reshape to [128]


        l5_weight = self.dropout(self.l5_weights(feature).view(256, 128))  # Reshape to [256, 128]
        l5_bias = self.dropout(self.l5_bias(feature).view(256) ) # Reshape to [256]


        l6_weight =self.dropout( self.l6_weights(feature).view(256, 256))  # Reshape to [256, 256]
        l6_bias = self.dropout(self.l6_bias(feature).view(256))

        for i, fc_layer_size in enumerate(self.fc_layer_sizes):
            if i == 0:
                output = F.linear(output, l4_weight, l4_bias)
            else:
                output = F.linear(output, l5_weight, l5_bias)


            output = F.layer_norm(output, normalized_shape=[self.fc_layer_sizes[i]], eps=1e-5)


            F.relu(output)

        output = F.linear(output, l6_weight, l6_bias)


        l1_weight = self.dropout(self.l1_weights(feature).view(self.task_dim, -1))
        l1_bias = self.dropout(self.l1_bias(feature).view(-1))

        output = F.linear(output, l1_weight, l1_bias)

        if self.task in ['tasks_binary_multilabel', 'mort_24h', 'mort_48h', 'LOS']:
            output = nn.Sigmoid()(output)

        # loss = self.loss(output, y.float())

        return output

    def init_ray(self, target_usr, f=1):
        if (f == 0):
            if self.dataset == "eicu":
                if (len(self.input_ray.data.shape) == 1):
                    if (len(self.usr_used) == 1):
                        self.input_ray.data[0] = 1.0
                    else:
                        for i in range(len(self.usr_used)):
                            if self.usr_used[i] == 7 or self.usr_used[i] == 8 or self.usr_used[i] == 9  :
                                self.input_ray.data[i] = 0.3
                            else:
                                self.input_ray.data[i] = 0
                elif (len(self.input_ray.data.shape) == 2):
                    if (len(self.usr_used) == 1):
                        self.input_ray.data[0, 0] = 1.0
                    else:
                        for i in range(len(self.usr_used)):
                            if self.usr_used[i] == 7 or self.usr_used[i] == 8 or self.usr_used[i] == 9:
                                self.input_ray.data[0, i] = 0.3
                            else:
                                self.input_ray.data[0, i] = 0

        else:
            if self.dataset == "eicu":
                if self.train_pt == True:
                    # if (len(self.input_ray.data.shape) == 1):
                    #     if (len(self.usr_used) == 1):
                    #         self.input_ray.data[0] = 1.0
                    #     else:
                    #         for i in range(len(self.usr_used)):
                    #             if self.usr_used[i] == 0 or self.usr_used[i] == 2:
                    #                 self.input_ray.data[i] = 0.5
                    #             elif self.usr_used[i] == 1 or self.usr_used[i] == 3 or self.usr_used == 4:
                    #                 self.input_ray.data[i] = 0.3
                    #             elif self.usr_used[i] == 6:
                    #                 self.input_ray.data[i] = 0.2
                    #             else:
                    #                 self.input_ray.data[i] = 0.1
                    #
                    # elif (len(self.input_ray.data.shape) == 2):
                    #     if (len(self.usr_used) == 1):
                    #         self.input_ray.data[0,0] = 1.0
                    #     else:
                    #         for i in range(len(self.usr_used)):
                    #             if self.usr_used[i] == 0 or self.usr_used[i] == 2:
                    #                 self.input_ray.data[0, i] = 0.5
                    #             elif self.usr_used[i] == 1 or self.usr_used[i] == 3 or self.usr_used == 4:
                    #                 self.input_ray.data[0, i] = 0.3
                    #             elif self.usr_used[i] == 6:
                    #                 self.input_ray.data[0, i] = 0.2
                    #             else:
                    #                 self.input_ray.data[0, i] = 0.1
                    if (len(self.input_ray.data.shape) == 1):
                        if (len(self.usr_used) == 1):
                            self.input_ray.data[0] = 1.0
                        else:
                            for i in range(len(self.usr_used)):
                                if self.usr_used[i] == 0 or self.usr_used[i] == 2:
                                    self.input_ray.data[i] = 0.5
                                elif self.usr_used[i] == 1 or self.usr_used[i] == 3 or self.usr_used[i] == 4:
                                    self.input_ray.data[i] = 0.5
                                elif self.usr_used[i] == 6:
                                    self.input_ray.data[i] = 0.2
                                else:
                                    self.input_ray.data[i] = 0.2

                    elif (len(self.input_ray.data.shape) == 2):
                        if (len(self.usr_used) == 1):
                            self.input_ray.data[0,0] = 1.0
                        else:
                            for i in range(len(self.usr_used)):
                                if self.usr_used[i] == 0 or self.usr_used[i] == 2:
                                    self.input_ray.data[0, i] = 0.5
                                elif self.usr_used[i] == 1 or self.usr_used[i] == 3 or self.usr_used[i]== 4:
                                    self.input_ray.data[0, i] = 0.5
                                elif self.usr_used[i] == 6:
                                    self.input_ray.data[0, i] = 0.2
                                else:
                                    self.input_ray.data[0, i] = 0.2
                else:
                    self.input_ray.data.fill_(1 / len(self.usr_used))
                    # if (len(self.input_ray.data.shape) == 1):
                    #     if (len(self.usr_used) == 1):
                    #         self.input_ray.data[0] = 1.0
                    #     else:
                    #         for i in range(len(self.usr_used)):
                    #             if i == target_usr:
                    #                 self.input_ray.data[i] = 0.8
                    #             else:
                    #                 self.input_ray.data[i] = (1.0 - 0.8) / (len(self.usr_used) - 1)
                    # elif (len(self.input_ray.data.shape) == 2):
                    #     if (len(self.usr_used) == 1):
                    #         self.input_ray.data[0, 0] = 1.0
                    #     else:
                    #         for i in range(len(self.usr_used)):
                    #             if i == target_usr:
                    #                 self.input_ray.data[0, i] = 0.8
                    #             else:
                    #                 self.input_ray.data[0, i] = (1.0 - 0.8) / (len(self.usr_used) - 1)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, norm_layer="ln", dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first


        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

    def forward(self, src, src_mask, src_key_padding_mask):

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class VGG(nn.Module):
    def __init__(self, features, num_classes=8, init_weights=False):
        super(VGG, self).__init__()
        self.featurs = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        # 初始化参数
        if init_weights:
            self._initialize_weights()

    def forward(self, x, input_ray=None):
        # N x 3 X 224 x224
        x = self.featurs(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 变量网络所有层
            if isinstance(m, nn.Conv2d):  # 是否为卷积层
                # 使用Kaiming初始化方法来初始化该层的权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 否具有偏差项
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 是否为Linear
                # 正太分布初始化全连接层
                nn.init.normal_(m.weight, 0, 0.01)
                # 将偏项设置为0
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    # 创建一个空列表用于存储神经网络的不同层
    layers = []
    # 初始输入通道数
    in_channels = 3
    # 遍历传入的配置列表
    for v in cfg:
        if v == "M":  # 池化层3
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            # 更新输入通道数，以便下一层的卷积层使用
            in_channels = v
    # 返回一个包含所有层的顺序容器，通常是一个特征提取器部分
    return nn.Sequential(*layers)


# 定义了不同VGG模型的卷积配置信息，其中 'M' 表示池化层，数字表示卷积层的输出通道数
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 定义一个函数 vgg，用于构建不同类型的VGG神经网络模型
def vgg(model_name="vgg16", **kwargs):
    # 检查传入的模型名是否在配置字典中
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    # 根据模型名获取对应的卷积配置信息
    cfg = cfgs[model_name]

    # 使用 make_features 函数创建特征提取器，然后将其传递给 VGG 模型
    model = VGG(make_features(cfg), **kwargs)
    return model

class SkinModel(nn.Module):
    def __init__(self, args, emb_type, pretrain=False, n_class=8, user_used=None, device=None):
        super(SkinModel, self).__init__()
        self.emb_type = emb_type
        self.pretrain = pretrain
        self.args = args
        hidden_dim = args.hidden_dim
        self.n_users = args.num_users
        self.train_pt = args.train_pt
        self.baseline_type = args.baseline_type
        self.usr_used = user_used
        self.input_ray = Variable(torch.FloatTensor([[1 / len(self.usr_used) for i in self.usr_used]])).to(device)
        self.input_ray.requires_grad = True
        spec_norm = args.spec_norm
        layers = [
            spectral_norm(nn.Linear(len(self.usr_used), hidden_dim)) if spec_norm else nn.Linear(len(self.usr_used),
                                                                                                 hidden_dim)]
        for _ in range(args.n_hidden - 1):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)


        if emb_type == "mobilenet":

            self.encoder = models.mobilenet_v3_small(pretrained=True)
            # self.encoder.classifier[1] = nn.Linear( self.encoder.classifier[1].in_features, n_class)
            # self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_class)
            # num_ftrs = self.encoder._fc.in_features
        # else:  # efficientnet with group normalization
        #     self.encoder = EfficientNet_GN.from_pretrained('efficientnet-b0', num_classes=n_class)
            # num_ftrs = self.encoder._fc.in_features
            # self.encoder._fc = nn.Linear(num_ftrs, n_class)

        self.n_class = n_class
        self.l1_weights = nn.Linear(hidden_dim, 1024*576)
        self.l1_bias = nn.Linear(hidden_dim, 1024)
        self.l2_weights = nn.Linear(hidden_dim, 8 * 1024)
        self.l2_bias = nn.Linear(hidden_dim, n_class)
        # self.l1_weights = nn.Linear(num_ftrs, n_class)
        # self.l1_bias = nn.Linear(n_class)


    def forward(self, img, input_ray=None):
        if input_ray != None:
            self.input_ray.data = input_ray

        feature = self.mlp(self.input_ray)
        l1_weight = self.l1_weights(feature).view(1024,-1)
        l1_bias = self.l1_bias(feature).view(-1)

        l2_weight = self.l2_weights(feature).view(8, -1)
        l2_bias = self.l2_bias(feature).view(-1)

        x = self.encoder.features(img)
        a = self.encoder.classifier[0]

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, l1_weight, l1_bias)
        x = F.linear(x,l2_weight,l2_bias)


        return x

    def init_ray(self, target_usr, f=1):
        if self.train_pt == True:
            if (len(self.input_ray.data.shape) == 1):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if self.usr_used[i] == 1 or self.usr_used[i] == 3:
                            self.input_ray.data[i] = 0.1
                        elif self.usr_used[i] == 0:
                            self.input_ray.data[i] = 0.5
                        elif self.usr_used[i] == 2:
                            self.input_ray.data[i] = 0.3
                        else:
                            self.input_ray.data[i] = 0.05
            elif (len(self.input_ray.data.shape) == 2):
                if (len(self.usr_used) == 1):
                    self.input_ray.data[0, 0] = 1.0
                else:
                    for i in range(len(self.usr_used)):
                        if self.usr_used[i] == 1 or self.usr_used[i] == 3:
                            self.input_ray.data[0,i] = 0.1
                        elif self.usr_used[i] == 0:
                            self.input_ray.data[0,i] = 0.5
                        elif self.usr_used[i] == 2:
                            self.input_ray.data[0,i] = 0.3
                        else:
                            self.input_ray.data[0,i] = 0.05
        else:
            self.input_ray.data.fill_(1 / len(self.usr_used))

class SkinModel1(nn.Module):
    def __init__(self, args, n_usrs, usr_used, device, n_classes=10, in_channels=3, n_kernels=8, hidden_dim=100,
                 spec_norm=False, n_hidden=1):
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.n_kernels = n_kernels
        self.n_classes = n_classes
        self.n_users = n_usrs
        self.usr_used = usr_used
        self.device = device

        self.input_ray = Variable(torch.FloatTensor([[1 / len(usr_used) for i in usr_used]])).to(device)
        self.input_ray.requires_grad = True

        layers = [
            spectral_norm(nn.Linear(len(usr_used), hidden_dim)) if spec_norm else nn.Linear(len(usr_used), hidden_dim)]

        for _ in range(2):
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        self.c1_weights = []
        self.c1_bias = []
        self.c2_weights = []
        self.c2_bias = []
        self.l1_weights = []
        self.l1_bias = []
        self.l2_weights = []
        self.l2_bias = []
        for _ in range(n_hidden - 1):
            self.c1_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c1_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.c2_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.c2_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l1_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l1_bias.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_weights.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_weights.append(nn.LeakyReLU(0.2, inplace=True))
            self.l2_bias.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim))
            self.l2_bias.append(nn.LeakyReLU(0.2, inplace=True))

        self.c1_weights = nn.Sequential(*(self.c1_weights + [
            spectral_norm(nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 7 * 7)) if spec_norm else nn.Linear(
                hidden_dim, self.n_kernels * self.in_channels * 7 * 7)]))
        self.c1_bias = nn.Sequential(*(self.c1_bias + [
            spectral_norm(nn.Linear(hidden_dim, self.n_kernels)) if spec_norm else nn.Linear(hidden_dim,
                                                                                             self.n_kernels)]))
        self.c2_weights = nn.Sequential(*(self.c2_weights + [spectral_norm(
            nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 7 * 7)) if spec_norm else nn.Linear(hidden_dim,
                                                                                                            2 * self.n_kernels * self.n_kernels * 7 * 7)]))
        self.c2_bias = nn.Sequential(*(self.c2_bias + [
            spectral_norm(nn.Linear(hidden_dim, 2 * self.n_kernels)) if spec_norm else nn.Linear(hidden_dim,
                                                                                                 2 * self.n_kernels)]))
        self.l1_weights = nn.Sequential(*(self.l1_weights + [
            spectral_norm(nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 7 * 7)) if spec_norm else nn.Linear(
                hidden_dim, 120 * 400)]))
        self.l1_bias = nn.Sequential(
            *(self.l1_bias + [spectral_norm(nn.Linear(hidden_dim, 120)) if spec_norm else nn.Linear(hidden_dim, 120)]))
        self.l2_weights = nn.Sequential(*(self.l2_weights + [
            spectral_norm(nn.Linear(hidden_dim, 84 * 120)) if spec_norm else nn.Linear(hidden_dim, 84 * 120)]))
        self.l2_bias = nn.Sequential(
            *(self.l2_bias + [spectral_norm(nn.Linear(hidden_dim, 84)) if spec_norm else nn.Linear(hidden_dim, 84)]))

        self.locals = nn.ModuleList([LocalOutput(n_output=n_classes) for i in range(self.n_users)])

    def forward(self, x, y, usr_id, input_ray=None):
        if input_ray != None:
            self.input_ray.data = input_ray.to(self.device)

        feature = self.mlp(self.input_ray)

        weights = {
            "conv1.weight": self.c1_weights(feature).view(self.n_kernels, self.in_channels, 7, 7),
            "conv1.bias": self.c1_bias(feature).view(-1),
            "conv2.weight": self.c2_weights(feature).view(2 * self.n_kernels, self.n_kernels, 7, 7),
            "conv2.bias": self.c2_bias(feature).view(-1),
            "fc1.weight": self.l1_weights(feature).view(120,400),
            "fc1.bias": self.l1_bias(feature).view(-1),
            "fc2.weight": self.l2_weights(feature).view(84, 120),
            "fc2.bias": self.l2_bias(feature).view(-1),
        }
        x = F.conv2d(x, weight=weights['conv1.weight'], bias=weights['conv1.bias'], stride=3)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, weight=weights['conv2.weight'], bias=weights['conv2.bias'], stride=3)
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(F.linear(x, weight=weights["fc1.weight"], bias=weights["fc1.bias"]), 0.2)
        logits = F.leaky_relu(F.linear(x, weight=weights["fc2.weight"], bias=weights["fc2.bias"]), 0.2)

        pred, loss = self.locals[usr_id](logits, y)

        return pred, loss