import copy
import math
import random
import time

from torch.utils.data import DataLoader

from hyper_model.benefit import DatasetSplit
from pfedgraph.test import compute_acc, compute_local_test_accuracy, compute_mse

import numpy as np
import torch
import torch.optim as optim


from pfedgraph.utils import aggregation_by_graph, update_graph_matrix_neighbor

from pfedgraph.attack import *


def local_train_pfedgraph(args,logger, round, nets_this_round, cluster_models, dataset_train, dataset_test, dict_users_train, dict_users_test, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list,device):
    dict_users_valid = dict()

    for usr in range(args.num_users):
        dict_users_valid[usr] = set(
            random.sample(set(dict_users_train[usr]), int(len(dict_users_train[usr]) / 5)))
        dict_users_train[usr] = set(dict_users_train[usr]) - dict_users_valid[usr]
    for net_id, net in nets_this_round.items():
        net.to(device)
        train_local_dls = [enumerate(
            DataLoader(DatasetSplit(dataset_train, dict_users_train[idxs]), batch_size=args.local_bs, shuffle=True,
                       num_workers=args.num_workers)) for idxs in range(args.num_users)]

        val_local_dls = [enumerate(
            DataLoader(DatasetSplit(dataset_train, dict_users_valid[idxs]), batch_size=args.local_bs, shuffle=True,
                       num_workers=args.num_workers)) for idxs in range(args.num_users)]

        test_dl = DataLoader(dataset=dataset_test, batch_size=args.local_bs, shuffle=False)
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                val_acc = compute_acc(net, val_local_dls[net_id],device)
                personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution,device)

                if val_acc > best_val_acc_list[net_id]:
                    best_val_acc_list[net_id] = val_acc
                    best_test_acc_list[net_id] = personalized_test_acc
                print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
                logger.info('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
            elif args.dataset=='synthetic1' or args.dataset=='synthetic2' or args.dataset=='synthetic3':
                test_local_dls = DataLoader(DatasetSplit(dataset_test, dict_users_test[net_id]),
                                            batch_size=len(dict_users_test[net_id]),
                                            shuffle=True,
                                            num_workers=args.num_workers)
                mse = compute_mse(net, test_local_dls, device)
                print(
                    '>> Client {} test1 | Mse: ({:.5f})'.format(
                        net_id, mse))
                logger.info(
                    '>> Client {} test1 | Mse: ({:.5f})'.format(
                        net_id, mse))


        # Set Optimizer
        if args.optim == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = optim.SGD( net.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        if round > 0:
            cluster_model = cluster_models[net_id].to(device)
        

        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.local_epochs):
            try:
                _,(x, target) = next(iterator)
            except StopIteration:
                train_local_dl = enumerate(
                    DataLoader(DatasetSplit(dataset_train, dict_users_train[net_id]), batch_size=args.local_bs,
                               shuffle=True,
                               num_workers=args.num_workers))
                iterator = iter(train_local_dl)
                _,(x, target) = next(iterator)
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
        

            if round > 0:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                val_local_dls = [enumerate(
                    DataLoader(DatasetSplit(dataset_train, dict_users_valid[idxs]), batch_size=args.local_bs, shuffle=True,
                               num_workers=args.num_workers)) for idxs in range(args.num_users)]

                test_dl = DataLoader(dataset=dataset_test, batch_size=args.local_bs, shuffle=False)
                val_acc = compute_acc(net, val_local_dls[net_id],device)
                personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution,device)

                if val_acc > best_val_acc_list[net_id]:
                    best_val_acc_list[net_id] = val_acc
                    best_test_acc_list[net_id] = personalized_test_acc
                print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
                logger.info('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
            elif args.dataset=='synthetic1' or args.dataset=='synthetic2' or args.dataset=='synthetic3':
                test_local_dls = DataLoader(DatasetSplit(dataset_test, dict_users_test[net_id]),
                                            batch_size=len(dict_users_test[net_id]),
                                            shuffle=True,
                                            num_workers=args.num_workers)
                mse = compute_mse(net, test_local_dls, device)
                print(
                    '>> Client {} test1 | Mse: ({:.5f})'.format(
                        net_id, mse))
                logger.info(
                    '>> Client {} test1 | Mse: ({:.5f})'.format(
                        net_id, mse))
            # net.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()



 