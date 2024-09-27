from collections import defaultdict
from re import T
import torch
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class eicuDatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, nrows = self.dataset[self.idxs[item]]
        x_nc = data[:, :, 8:-1]
        x_cat = data[:, :, 1:8].astype(int)
        label = data[:,:,-1]
        return x_nc, x_cat, label


class data(object):
    def __init__(self, args, dataset_train=None, dataset_test=None, dict_user_train=None, dict_user_test=None,
                 users_used=None,num_batches=None,traindata_cls_counts=None):
        if args.dataset == 'skin':
            self.num_batches = num_batches
            self.dict_user_train = dict_user_train
            self.traindata_cls_counts = traindata_cls_counts
        else:
            self.target_usr = args.target_usr
            if users_used == None:
                users_used = [i for i in range(self.args.num_users)]
            self.users_used = users_used
            self.traindata_cls_counts = traindata_cls_counts
            dict_user_valid = dict()
            # if args.dataset == 'eicu':
            #     for usr in users_used:
            #         dict_user_train[usr] =  set(dict_user_train[usr])
            #         dict_user_test[usr] = set(dict_user_test[usr])
            # else:
            if args.train_baseline:
                for usr in users_used:
                    # dict_user_train[usr] =  set(dict_user_train[usr])
                    dict_user_valid[usr] = set(
                        random.sample(set(dict_user_train[usr]), int(len(dict_user_train[usr]) / 10)))  # 1/10作为验证集
                    dict_user_train[usr] = set(dict_user_train[usr]) - dict_user_valid[usr]
            else:
                for usr in users_used:
                    dict_user_valid[usr] = set(random.sample(set(dict_user_train[usr]), int(len(dict_user_train[usr]) / 10)))#1/10作为验证集
                    dict_user_train[usr] = set(dict_user_train[usr]) -  dict_user_valid[usr]
                    # dict_user_valid[usr] = set(random.sample(set(dict_user_train[usr]), 1))
                    # dict_user_train[usr] = set(dict_user_train[usr]) -  dict_user_valid[usr]
            # self.data_valid = DataLoader(DatasetSplit(dataset_train, dict_user_valid), batch_size=len(dict_user_valid), shuffle=False)
            self.dataset_test = dataset_test
            self.dict_user_test = dict_user_test
            self.dataset_train = dataset_train
            self.data_test = dataset_test
            self.dict_user_train = dict_user_train
            self.dict_user_valid = dict_user_valid
            self.num_batches = defaultdict(list)
            self.traindata_cls_counts = traindata_cls_counts



            if args.local_bs == -1:
                self.train_loaders = [enumerate(
                    DataLoader(DatasetSplit(dataset_train, dict_user_train[idxs]), batch_size=len(dict_user_train[idxs]),
                               shuffle=True, num_workers=args.num_workers)) for idxs in range(args.num_users)]
                for i in range(args.num_users):
                    self.num_batches[i] = len(DataLoader(DatasetSplit(dataset_train, dict_user_train[i]), batch_size=len(dict_user_train[i]),
                               shuffle=True, num_workers=args.num_workers))
            else:
                self.train_loaders = [enumerate(
                    DataLoader(DatasetSplit(dataset_train, dict_user_train[idxs]), batch_size=args.local_bs, shuffle=True,
                               num_workers=args.num_workers,drop_last=True)) for idxs in range(args.num_users)]
                for i in range(args.num_users):
                    self.num_batches[i] = len(DataLoader(DatasetSplit(dataset_train, dict_user_train[i]), batch_size=args.local_bs,
                               shuffle=True, drop_last=False,num_workers=args.num_workers))
            if args.train_baseline:
                # self.valid_loaders = None
                self.valid_loaders = [enumerate(
                    DataLoader(DatasetSplit(dataset_train, dict_user_valid[idxs]),
                               batch_size=len(dict_user_valid[idxs]),
                               shuffle=True, num_workers=args.num_workers)) for idxs in range(args.num_users)]
            else:
                self.valid_loaders = [enumerate(
                    DataLoader(DatasetSplit(dataset_train, dict_user_valid[idxs]), batch_size=len(dict_user_valid[idxs]),
                            shuffle=True, num_workers=args.num_workers)) for idxs in range(args.num_users)]

            self.test_loaders = [enumerate(DataLoader(DatasetSplit(dataset_test, dict_user_test[idxs]),
                                                     batch_size=len(dict_user_test[idxs]), shuffle=True,
                                                     num_workers=args.num_workers))
                            for idxs in range(args.num_users)]

      