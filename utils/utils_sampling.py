import math
import random
from itertools import permutations
import numpy as np
import torch
import pdb

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def shuffle_and_ensure_unique_pairs(lst, pair_size):

    assert len(lst) % pair_size == 0, "List length must be a multiple of the pair size."

    half_size = len(lst) // 2
    first_half = lst[:half_size]
    second_half = lst[half_size:]

    random.shuffle(first_half)
    random.shuffle(second_half)

    shuffled_list = [val for pair in zip(first_half, second_half) for val in pair]

    return shuffled_list

def shuffle_and_ensure_unique_pairs1(lst, pair_size):
    assert len(lst) % pair_size == 0, "List length must be a multiple of the pair size."

    # 分成pair_size份
    segment_size = len(lst) // pair_size
    segments = [lst[i*segment_size:(i+1)*segment_size] for i in range(pair_size)]

    # 分别随机打乱每一份
    for segment in segments:
        random.shuffle(segment)

    # 按顺序重新组合这些份
    shuffled_list = []
    for i in range(segment_size):
        for segment in segments:
            shuffled_list.append(segment[i])

    return shuffled_list
def noniid(dataset, num_users, shard_per_user, partition, rand_set_all=[]):
    if partition == "labeldir":
        train_label = np.array(dataset.targets)
        n_train = len(train_label)
        K = int(train_label.max() + 1)  # 类别数
        net_dataidx_map = {}
        min_size = 0
        min_require_size = 10  # 每个用户的最小数据点数
        beta = 0.5  # 狄利克雷分布的浓度参数 0.5

        if len(rand_set_all) != 0:
            sample_per_user = n_train / num_users
            new_class_dis = np.zeros_like(rand_set_all)
            for i in range(num_users):
                total = rand_set_all[i].sum()
                proportions =  rand_set_all[i] / total
                new_class_dis[i] = np.round(proportions*sample_per_user)

            net_dataidx_map = {}
            for user in range(num_users):
                net_dataidx_map[user] = []

            for k in range(K):
                idx_k = np.where(train_label==k)[0]
                np.random.shuffle(idx_k)
                start = 0
                for user in range(num_users):
                    end = start + int(new_class_dis[user,k])
                    net_dataidx_map[user].extend(idx_k[start:end].tolist())
                    start = end
            class_dis = rand_set_all
        else:
            while min_size < min_require_size:
                idx_batch = [[] for _ in range(num_users)]
                for k in range(K):
                    idx_k = np.where(train_label == k)[0]
                    np.random.shuffle(idx_k)

                    proportions = np.random.dirichlet(np.repeat(beta, num_users))
                    proportions = np.array(
                        [p * (len(idx_j) < n_train / num_users) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min(len(idx_j) for idx_j in idx_batch)

            for j in range(num_users):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            class_dis = np.zeros((num_users, K))
            for j in range(num_users):
                for m in range(K):
                    class_dis[j, m] = int((train_label[net_dataidx_map[j]] == m).sum())
        return net_dataidx_map, class_dis
    if partition == 'diff-dir':
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {label: np.where(np.array(dataset.targets) == label)[0] for label in
                     range(len(np.unique(dataset.targets)))}

        # 初始化记录每个用户获得的标签比例的矩阵
        rand_set_all = np.zeros((num_users,len(np.unique(dataset.targets))))

        # 按狄利克雷分布分配每个标签的索引
        for label, idxs in idxs_dict.items():
            idxs = np.array(idxs)
            np.random.shuffle(idxs)

            # 生成狄利克雷分布
            proportions = np.random.dirichlet(np.repeat(0.5, num_users))

            # 更新 rand_set_all 来记录每个用户获得的标签比例
            rand_set_all[:, label] = proportions

            # 确保比例总和为1
            proportions = proportions / proportions.sum()

            # 计算每个用户应分配的索引数量
            proportions = np.cumsum(proportions) * len(idxs)
            proportions = np.round(proportions).astype(int)[:-1]

            # 分割索引并分配给用户
            idxs_split = np.split(idxs, proportions)
            for user in range(num_users):
                if user < len(idxs_split):
                    dict_users[user] = np.concatenate((dict_users[user], idxs_split[user]), axis=0)

        # 验证分配
        all_idxs = np.concatenate(list(dict_users.values()))
        assert len(all_idxs) == len(dataset), "Not all examples are distributed."
        assert len(set(all_idxs)) == len(dataset), "Some examples are duplicated."

        return dict_users, rand_set_all
    if partition == 'kdd':
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

        idxs_dict = {}
        for i in range(len(dataset)):
            label = torch.tensor(dataset.targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

        num_classes = len(np.unique(dataset.targets))
        shard_per_class = int(shard_per_user * num_users / num_classes)
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x

        if len(rand_set_all) == 0:
            rand_set_all = list(range(num_classes)) * shard_per_class
            rand_set_all = sorted(rand_set_all)
            # random.shuffle(rand_set_all)
            if shard_per_user != 2:
                rand_set_all = shuffle_and_ensure_unique_pairs1(rand_set_all, shard_per_user)
            else:
                rand_set_all = shuffle_and_ensure_unique_pairs(rand_set_all, shard_per_user)
            rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

        # divide and assign
        for i in range(num_users):
            rand_set_label = rand_set_all[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            dict_users[i] = np.concatenate(rand_set)

        test = []
        for key, value in dict_users.items():
            x = np.unique(torch.tensor(dataset.targets)[value])
            assert(len(x)) <= shard_per_user
            test.append(value)
        test = np.concatenate(test)
        assert(len(test) == len(dataset))
        assert(len(set(list(test))) == len(dataset))
        # print(dict_users[0], rand_set_all)
        return dict_users, rand_set_all


