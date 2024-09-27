# code for data prepare
from collections import defaultdict

from PIL import Image
from torch.utils.data import RandomSampler, DataLoader, Dataset
from torchvision import datasets, transforms

from hyper_model.cached_dataset import CachedDataset, CachedDataset1
from utils.utils_sampling import iid, noniid
import os
import pdb 
import json
import numpy as np 
import torch
import random


trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])


def simple_data(args):

    dataset_train = []
    dataset_test = []
    dict_users_train = {}
    dict_users_test = {}


    if args.dataset == "synthetic1":

        args.num_users = 8
        v0 = np.random.random((args.input_dim,))
        v1 = np.random.random((args.input_dim,))
        v2 = np.random.random((args.input_dim,))
        mean = np.zeros((args.input_dim,))
        cov = args.std ** 2 * np.eye(args.input_dim)

        for usr in range(8):

            if(usr in [0, 1, 4, 5]):
                tmp_trainN = 2000
                tmp_testN = 2000
            else:
                tmp_trainN = 100
                tmp_testN = 100

            r_0 = np.random.multivariate_normal(mean, cov)
            r_1 = np.random.multivariate_normal(mean, cov)
            r_2 = np.random.multivariate_normal(mean, cov)
            u_m_0 = v0 + r_0
            u_m_1 = v1 + r_1
            u_m_2 = v2 + r_2
            x_m = np.random.uniform(-1.0, 1.0, (tmp_trainN + tmp_testN, args.input_dim))
            x_0 = x_m ** 3
            x_1 = x_m ** 2
            x_2 = x_m ** 1
            y_m = np.dot(x_0, u_m_0) + np.dot(x_1, u_m_1) + np.dot(x_2,u_m_2) + np.random.normal(0, args.sigma ** 2, (tmp_trainN + tmp_testN,))#y_m = u_m_0*x_m^3+u_m_1*x_m^2+u_m_2*x_m^1+keci
            dataset_train.extend([(x.astype(np.float32), y) for x, y in zip(x_m[:tmp_trainN], y_m[:tmp_trainN])])
            dataset_test.extend([(x.astype(np.float32), y) for x, y in zip(x_m[-tmp_testN:], y_m[-tmp_testN:])])

            try:
                dict_users_train[usr] = [i+ dict_users_train[usr-1][-1]+1 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i+ dict_users_test[usr-1][-1]+1 for i in range(tmp_testN)]
            except KeyError:
                dict_users_train[usr] = [i+ 0  for i in range(tmp_trainN)]
                dict_users_test[usr] = [i+ 0  for i in range(tmp_testN)]

    elif args.dataset == "synthetic2":

        args.num_users = 8
        v0 = np.random.random((args.input_dim,))
        v1 = np.random.random((args.input_dim,))
        v2 = np.random.random((args.input_dim,))
        mean = np.zeros((args.input_dim,))
        cov = args.std ** 2 * np.eye(args.input_dim)

        for usr in range(8):

            tmp_trainN = 2000
            tmp_testN = 2000

            r_0 = np.random.multivariate_normal(mean, cov)
            r_1 = np.random.multivariate_normal(mean, cov)
            r_2 = np.random.multivariate_normal(mean, cov)
            u_m_0 = v0 + r_0
            u_m_1 = v1 + r_1
            u_m_2 = v2 + r_2
            x_m = np.random.uniform(-1.0, 1.0, (tmp_trainN + tmp_testN, args.input_dim))
            x_0 = x_m ** 3
            x_1 = x_m ** 2
            x_2 = x_m ** 1

            if usr in [0,1,2,3]:
                y_m = np.dot(x_0, u_m_0) + np.dot(x_1, u_m_1) + np.dot(x_2, u_m_2) + np.random.normal(0, args.sigma ** 2, (tmp_trainN + tmp_testN,))
            else:
                y_m = - (np.dot(x_0, u_m_0) + np.dot(x_1, u_m_1) + np.dot(x_2, u_m_2) + np.random.normal(0,args.sigma ** 2, (tmp_trainN + tmp_testN,)))

            dataset_train.extend([(x.astype(np.float32), y) for x, y in zip(x_m[:tmp_trainN], y_m[:tmp_trainN])])
            dataset_test.extend([(x.astype(np.float32), y) for x, y in zip(x_m[-tmp_testN:], y_m[-tmp_testN:])])

            try:
                dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in range(tmp_testN)]
            except KeyError:
                dict_users_train[usr] = [i + 0 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i + 0 for i in range(tmp_testN)]

    elif args.dataset == "synthetic3":

        args.num_users = 8
        v0 = np.random.random((args.input_dim,))
        v1 = np.random.random((args.input_dim,))
        v2 = np.random.random((args.input_dim,))
        mean = np.zeros((args.input_dim,))
        cov = args.std ** 2 * np.eye(args.input_dim)

        for usr in range(8):

            tmp_trainN = 2000
            tmp_testN = 2000

            r_0 = np.random.multivariate_normal(mean, cov)
            r_1 = np.random.multivariate_normal(mean, cov)
            r_2 = np.random.multivariate_normal(mean, cov)
            u_m_0 = v0 + r_0
            u_m_1 = v1 + r_1
            u_m_2 = v2 + r_2
            x_m = np.random.uniform(-0.5, 0.5, (tmp_trainN + tmp_testN, args.input_dim))
            x_0 = x_m ** 3
            x_1 = x_m ** 2
            x_2 = x_m ** 1

            if usr in [0,1,2,3]:
                theta = np.dot(x_0, u_m_0) + np.dot(x_1, u_m_1) + np.dot(x_2, u_m_2)
                y_m = l1(theta) + np.random.normal(0, args.sigma ** 2, (tmp_trainN + tmp_testN,))
            else:
                theta = np.dot(x_0, u_m_0) + np.dot(x_1, u_m_1) + np.dot(x_2, u_m_2)
                y_m = -l2(theta) + np.random.normal(0, args.sigma ** 2, (tmp_trainN + tmp_testN,))

            dataset_train.extend([(x.astype(np.float32), y) for x, y in zip(x_m[:tmp_trainN], y_m[:tmp_trainN])])
            dataset_test.extend([(x.astype(np.float32), y) for x, y in zip(x_m[-tmp_testN:], y_m[-tmp_testN:])])

            try:
                dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in range(tmp_testN)]
            except KeyError:
                dict_users_train[usr] = [i + 0 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i + 0 for i in range(tmp_testN)]

    elif args.dataset == "synthetic4":

        args.num_users = 8
        v0 = np.random.random((args.input_dim,))
        v1 = np.random.random((args.input_dim,))
        v2 = np.random.random((args.input_dim,))
        mean = np.zeros((args.input_dim,))
        cov = args.std ** 2 * np.eye(args.input_dim)

        for usr in range(8):

            tmp_trainN = 2000
            tmp_testN = 2000

            # Different distributions for different user groups
            if usr < 4:  # Users 0 to 3
                Zi = np.random.normal(1, 1, (args.input_dim,))
            else:  # Users 4 to 7
                Zi = np.random.normal(10, 1, (args.input_dim,))

            # Normalize Zi to get Xi
            Xi = Zi / np.linalg.norm(Zi)

            # Compute the random noise for the user model
            r_0 = np.random.multivariate_normal(np.zeros((args.input_dim,)), args.std ** 2 * np.eye(args.input_dim))
            r_1 = np.random.multivariate_normal(np.zeros((args.input_dim,)), args.std ** 2 * np.eye(args.input_dim))
            r_2 = np.random.multivariate_normal(np.zeros((args.input_dim,)), args.std ** 2 * np.eye(args.input_dim))


            u_m_0 = v0 + r_0
            u_m_1 = v1 + r_1
            u_m_2 = v2 + r_2


            y_m = np.dot(Xi ** 3, u_m_0) + np.dot(Xi ** 2, u_m_1) + np.dot(Xi, u_m_2) + np.random.normal(0,
                                                                                                         args.sigma ** 2)


            y_m = np.repeat(y_m, tmp_trainN + tmp_testN)


            if usr >= 4:
                y_m = -np.dot(Xi ** 3, u_m_0) + np.dot(Xi ** 2, u_m_1) + np.dot(Xi, u_m_2) + np.random.normal(0,
                                                                                                         args.sigma ** 2)


            Xi_extended = np.tile(Xi, (tmp_trainN + tmp_testN, 1))


            dataset_train.extend(
                [(x.astype(np.float32), y) for x, y in zip(Xi_extended[:tmp_trainN], y_m[:tmp_trainN])])
            dataset_test.extend([(x.astype(np.float32), y) for x, y in zip(Xi_extended[-tmp_testN:], y_m[-tmp_testN:])])

            try:
                dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in range(tmp_trainN)]
                dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in range(tmp_testN)]
            except KeyError:
                dict_users_train[usr] = [i for i in range(tmp_trainN)]
                dict_users_test[usr] = [i for i in range(tmp_testN)]


    return dataset_train, dataset_test, dict_users_train, dict_users_test

def l1(theta):
    return 1 - np.exp(-np.linalg.norm(theta - 1/2)**2)

def l2(theta):
    return 1 - np.exp(-np.linalg.norm(theta + 1/2)**2)
def nan_checker(args, patients):
    new_patients = []

    for patient in patients:
        temp = CachedDataset1("total", [patient], args.data_path)

        if args.task == "mort_24h" or args.task == "mort_48h" or args.task == "LOS":
            index = ["mort_24h", "mort_48h", "LOS"].index(args.task)
            label = temp[0]['tasks_binary_multilabel'][index]
        else:
            label = temp[0][args.task]

        check = torch.isnan(label).item()

        if check == False:
            new_patients.append(patient)

    return new_patients


def get_dataset(args):
    new_dir = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/data_split_fixed"
    client_id = args.hospital_id

    train_dataset, valid_dataset, test_dataset = [], [], []
    dict_users_train = {}
    dict_users_test = {}
    dict_users_valid = {}
    usr = 0

    for c_id in client_id:
        each_client = str(c_id)

        ##########################################################
        if args.task in ['mort_24h', 'mort_48h', 'LOS']:
            with open(os.path.join(new_dir, f"{c_id}.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']
        else:
            with open(os.path.join(new_dir, f"{c_id}_ver2.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']

        ##########################################################

        print(f"{c_id} client's patient number : ", len(train_patients + valid_patients + test_patients))

        train_patients = nan_checker(args, train_patients)
        valid_patients = nan_checker(args, valid_patients)
        test_patients = nan_checker(args, test_patients)


        train_dataset.extend([(x,y) for x,y in CachedDataset(each_client, train_patients, args.data_path,args.task)])
        valid_dataset.extend([(x,y) for x,y in CachedDataset(each_client, valid_patients, args.data_path,args.task)])
        test_dataset.extend([(x,y) for x,y in CachedDataset(each_client, test_patients, args.data_path,args.task)])

        try:
            dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in range(len(train_dataset)-dict_users_train[usr-1][-1]-1)]
            dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in range(len(test_dataset)-dict_users_test[usr-1][-1]-1)]
            dict_users_valid[usr] = [i + dict_users_valid[usr - 1][-1] + 1 for i in range(len(valid_dataset)-dict_users_valid[usr-1][-1]-1)]
        except KeyError:
            dict_users_train[usr] = [i + 0 for i in range(len(train_dataset))]
            dict_users_test[usr] = [i + 0 for i in range(len(test_dataset))]
            dict_users_valid[usr] = [i + 0 for i in range(len(valid_dataset))]

        usr = usr + 1
    labels = []
    for item in train_dataset:
        label = int(item[1].item())
        labels.append(label)

    traindata_cls_counts = record_net_data_stats(np.array(labels), dict_users_train)
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:, np.newaxis]


    return client_id, train_dataset, valid_dataset, test_dataset,dict_users_train,dict_users_test,dict_users_valid

def get_dataset1(args):
    new_dir = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/data_split_fixed"
    client_id = args.hospital_id

    train_dataset, valid_dataset, test_dataset = [], [], []


    for c_id in client_id:
        each_client = str(c_id)

        ##########################################################
        if args.task in ['mort_24h', 'mort_48h', 'LOS']:
            with open(os.path.join(new_dir, f"{c_id}.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']
        else:
            with open(os.path.join(new_dir, f"{c_id}_ver2.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']

        ##########################################################

        print(f"{c_id} client's patient number : ", len(train_patients + valid_patients + test_patients))

        train_patients = nan_checker(args, train_patients)
        valid_patients = nan_checker(args, valid_patients)
        test_patients = nan_checker(args, test_patients)

        train_dataset.append( CachedDataset1(each_client, train_patients, args.data_path))
        valid_dataset.append( CachedDataset1(each_client, valid_patients, args.data_path))
        test_dataset.append( CachedDataset1(each_client, test_patients, args.data_path))

    return client_id, train_dataset, valid_dataset, test_dataset
def get_dataloader(args, train_dataset, valid_dataset, test_dataset):

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    for i in range(len(args.hospital_id)) :
        client_weights.append( len(train_dataset[i]) )

        train_sampler = RandomSampler(train_dataset[i])
        if len(train_dataset[i]) % args.batch_size == 1:
            train_loaders.append( DataLoader(train_dataset[i], sampler=train_sampler, batch_size=args.batch_size, num_workers=0, drop_last=True) )
        else :
            train_loaders.append( DataLoader(train_dataset[i], sampler=train_sampler, batch_size=args.batch_size, num_workers=0, drop_last=False) )


        valid_sampler = RandomSampler(valid_dataset[i])
        if len(valid_dataset[i]) % args.batch_size == 1 :
            valid_loaders.append( DataLoader(valid_dataset[i], sampler=valid_sampler, batch_size=args.batch_size, num_workers=0, drop_last=True) )
        else :
            valid_loaders.append( DataLoader(valid_dataset[i], sampler=valid_sampler, batch_size=args.batch_size, num_workers=0, drop_last=False) )

        if len(test_dataset[i]) % args.batch_size == 1 :
            test_loaders.append( DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=0, drop_last=True) )
        else :
            test_loaders.append( DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=0, drop_last=False) )


    total = sum(client_weights)
    client_weights = [ weight / total for weight in client_weights ]

    return client_weights, train_loaders, valid_loaders, test_loaders

def get_data(args):
    data_distributions =None
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(args.data_root, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(args.data_root, train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,args.partition)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.partition,rand_set_all=rand_set_all)

        train_label = np.array(dataset_train.targets)
        traindata_cls_counts = record_net_data_stats(train_label, dict_users_train)
        data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:, np.newaxis]
    
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_root, "cifar10"), train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(os.path.join(args.data_root, "cifar10"), train=False, download=True, transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,args.partition)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user,args.partition, rand_set_all=rand_set_all)

        train_label = np.array(dataset_train.targets)
        traindata_cls_counts = record_net_data_stats(train_label, dict_users_train)
        data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:, np.newaxis]


    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(os.path.join(args.data_root, "cifar100"), train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR100(os.path.join(args.data_root, "cifar100"), train=False, download=True, transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,args.partition)
            # print(len(dict_users_train[0]),rand_set_all)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user,args.partition, rand_set_all=rand_set_all)

            train_label = np.array(dataset_train.targets)
            traindata_cls_counts = record_net_data_stats(train_label, dict_users_train)
            data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:, np.newaxis]

            test_label = np.array(dataset_test.targets)
            testdata_cls_counts = record_net_data_stats(test_label, dict_users_test)
            data_distributions_test = testdata_cls_counts / testdata_cls_counts.sum(axis=1)[:, np.newaxis]
            # print(rand_set_all)
            # exit()

    elif args.dataset == "synthetic1" or args.dataset == "synthetic2" or args.dataset=='synthetic3':
        
        dataset_train, dataset_test, dict_users_train, dict_users_test = simple_data(args)
        data_distributions = None


    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users_train, dict_users_test,data_distributions,traindata_cls_counts


class SkinCancerDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item]['img_path']
        image_path = image_path.replace("\\","/")
        image_path = image_path.replace("FCl", "FCL")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(self.data[item]['extended_labels'].index(1.0))

        return image, label

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))


    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts_dict))

    print(net_cls_counts_npy.astype(int))
    return net_cls_counts_npy
def get_dataloader_skin(args, total_data,train_label) :

    train_data = total_data['train']
    valid_data = total_data['valid']
    test_data = total_data['test']


    ################################################################################################

    train_dataset = SkinCancerDataset(train_data, transform=train_transform)
    valid_dataset = SkinCancerDataset(valid_data, transform=test_transform)
    test_dataset = SkinCancerDataset(test_data, transform=test_transform)
    # for x,y in train_dataset:
    #     train_label.append(y)
    if len(train_dataset) % args.batch_size == 1 :
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=True)
    else :
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=False)

    if len(valid_dataset) % args.batch_size == 1 :
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=True)
    else :
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=False)

    if len(test_dataset) % args.batch_size == 1 :
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=True)
    else :
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=False)

    return train_dataloader, valid_dataloader, test_dataloader,train_label
def get_federated_dataset(args):

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []
    dict_users_train = {}
    dict_users_test = {}
    dict_users_valid = {}
    num_batches = defaultdict(list)
    usr = 0
    train_label = []
    traindata_cls_counts = None
    print("### LOAD federated dataset ###")

    if "barcelona" in args.clients :
        barcelona_path = os.path.join( f"{args.data_path}/ISIC_2019", "ISIC_19_Barcelona_split.json")
        with open(barcelona_path, 'r') as f:
            barcelona_data = json.load(f)



        train_dataloader, valid_dataloader, test_dataloader,train_label = get_dataloader_skin(args, barcelona_data,train_label)
        if usr == 0 :
            dict_users_train[usr] = [i + 0 for i in range(len(barcelona_data['train']))]
            dict_users_test[usr] = [i + 0 for i in range(len(barcelona_data['test']))]
            dict_users_valid[usr] = [i + 0 for i in range(len(barcelona_data['valid']))]
            num_batches[usr] = len(train_dataloader)
            usr = usr +1
        else:
            dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in
                                     range(len(barcelona_data['train']))]
            dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in
                                    range(len(barcelona_data['test']) )]
            dict_users_valid[usr] = [i + dict_users_valid[usr - 1][-1] + 1 for i in
                                     range(len(barcelona_data['valid']))]

            num_batches[usr] = len(train_dataloader)
            usr = usr + 1

        client_weights.append( len(barcelona_data['train'] + barcelona_data['valid'] + barcelona_data['test'] ) ) # 0.7
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "rosendahl" in args.clients :
        rosendahl_path = os.path.join( f"{args.data_path}/HAM10000", "HAM_rosendahl_split.json")
        with open(rosendahl_path, 'r') as f:
            rosendahl_data = json.load(f)



        train_dataloader, valid_dataloader, test_dataloader,train_label = get_dataloader_skin(args, rosendahl_data,train_label)
        if usr == 0 :
            dict_users_train[usr] = [i + 0 for i in range(len(rosendahl_data['train']))]
            dict_users_test[usr] = [i + 0 for i in range(len(rosendahl_data['test']))]
            dict_users_valid[usr] = [i + 0 for i in range(len(rosendahl_data['valid']))]
            num_batches[usr] = len(train_dataloader)
            usr = usr +1
        else:
            dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in
                                     range(len(rosendahl_data['train']))]
            dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in
                                    range(len(rosendahl_data['test']) )]
            dict_users_valid[usr] = [i + dict_users_valid[usr - 1][-1] + 1 for i in
                                     range(len(rosendahl_data['valid']))]

            num_batches[usr] = len(train_dataloader)
            usr = usr + 1

        client_weights.append( len(rosendahl_data['train'] + rosendahl_data['valid'] + rosendahl_data['test']  )) # 0.7
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "vienna" in args.clients :
        vienna_path = os.path.join( f"{args.data_path}/HAM10000", "HAM_vienna_split.json")
        with open(vienna_path, 'r') as f:
            vienna_data = json.load(f)


        train_dataloader, valid_dataloader, test_dataloader,train_label = get_dataloader_skin(args, vienna_data,train_label)
        if usr == 0 :
            dict_users_train[usr] = [i + 0 for i in range(len(vienna_data['train']))]
            dict_users_test[usr] = [i + 0 for i in range(len(vienna_data['test']))]
            dict_users_valid[usr] = [i + 0 for i in range(len(vienna_data['valid']))]
            num_batches[usr] = len(train_dataloader)
            usr = usr +1
        else:
            dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in
                                     range(len(vienna_data['train']))]
            dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in
                                    range(len(vienna_data['test']) )]
            dict_users_valid[usr] = [i + dict_users_valid[usr - 1][-1] + 1 for i in
                                     range(len(vienna_data['valid']))]

            num_batches[usr] = len(train_dataloader)
            usr = usr + 1

        client_weights.append( len(vienna_data['train'] + vienna_data['valid'] + vienna_data['test'] ) )
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "PAD_UFES_20" in args.clients :
        PAD_UFES_20_path = os.path.join(f"{args.data_path}/PAD-UFES-20", "PAD_UFES_20_split.json")
        with open(PAD_UFES_20_path, 'r') as f:
            PAD_UFES_20_data = json.load(f)


        train_dataloader, valid_dataloader, test_dataloader,train_label = get_dataloader_skin(args, PAD_UFES_20_data,train_label)
        if usr == 0 :
            dict_users_train[usr] = [i + 0 for i in range(len(PAD_UFES_20_data['train']))]
            dict_users_test[usr] = [i + 0 for i in range(len(PAD_UFES_20_data['test']))]
            dict_users_valid[usr] = [i + 0 for i in range(len(PAD_UFES_20_data['valid']))]
            num_batches[usr] = len(train_dataloader)
            usr = usr +1
        else:
            dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in
                                     range(len(PAD_UFES_20_data['train']))]
            dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in
                                    range(len(PAD_UFES_20_data['test']) )]
            dict_users_valid[usr] = [i + dict_users_valid[usr - 1][-1] + 1 for i in
                                     range(len(PAD_UFES_20_data['valid']))]

            num_batches[usr] = len(train_dataloader)
            usr = usr + 1

        client_weights.append( len(PAD_UFES_20_data['train'] + PAD_UFES_20_data['valid'] + PAD_UFES_20_data['test'] ) )
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    if "Derm7pt" in args.clients :
        Derm7pt_path = os.path.join(f"{args.data_path}/Derm7pt", "Derm7pt_split.json")
        with open(Derm7pt_path, 'r') as f:
            Derm7pt_data = json.load(f)



        train_dataloader, valid_dataloader, test_dataloader,train_label = get_dataloader_skin(args, Derm7pt_data,train_label)
        if usr == 0 :
            dict_users_train[usr] = [i + 0 for i in range(len(Derm7pt_data['train']))]
            dict_users_test[usr] = [i + 0 for i in range(len(Derm7pt_data['test']))]
            dict_users_valid[usr] = [i + 0 for i in range(len(Derm7pt_data['valid']))]
            num_batches[usr] = len(train_dataloader)
            usr = usr +1
        else:
            dict_users_train[usr] = [i + dict_users_train[usr - 1][-1] + 1 for i in
                                     range(len(Derm7pt_data['train']))]
            dict_users_test[usr] = [i + dict_users_test[usr - 1][-1] + 1 for i in
                                    range(len(Derm7pt_data['test']) )]
            dict_users_valid[usr] = [i + dict_users_valid[usr - 1][-1] + 1 for i in
                                     range(len(Derm7pt_data['valid']))]

            num_batches[usr] = len(train_dataloader)
            usr = usr + 1

        client_weights.append( len(Derm7pt_data['train'] + Derm7pt_data['valid'] + Derm7pt_data['test'] ) )
        train_loaders.append(train_dataloader)
        valid_loaders.append(valid_dataloader)
        test_loaders.append(test_dataloader)

    total_size = sum(client_weights)
    print(client_weights)
    client_weights = [ float(c / total_size) for c in client_weights]

    # traindata_cls_counts = record_net_data_stats(np.array(train_label), dict_users_train)
    # data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:, np.newaxis]
    # print(traindata_cls_counts)



    return train_loaders, valid_loaders, test_loaders, client_weights,dict_users_train,num_batches,traindata_cls_counts


