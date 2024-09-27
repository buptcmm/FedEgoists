import argparse
import collections
import copy
import json
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from tqdm import trange
import pdb
from torch.utils.data import DataLoader, Dataset
from hyper_model.models import Hypernet, HyperSimpleNet, SimpleNet, Basenet_cifar, TransformerModel, SkinModel, vgg, \
    SkinModel1, TransformerModel1, CNNHyper100, TransformerHyper, TransformerTarget
from hyper_model.solvers import EPOSolver, LinearScalarizationSolver
from torch.autograd import Variable
import os
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import time
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

"""
training a personalized model for each client;
"""

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


from scipy import special


def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients), :]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:, np.newaxis]

    if metric == 'cosine':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                    np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric == 'only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution) / (
                    np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores > 0.9, 0.01, float('inf'))
    elif metric == 'l1':
        difference = np.linalg.norm(local_distributions - hypo_distribution, ord=1, axis=1)
    elif metric == 'l2':
        difference = np.linalg.norm(local_distributions - hypo_distribution, axis=1)
    elif metric == 'kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(
            difference)
    return difference


def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a * distribution_difference + b

    if np.sum(weight_tmp > 0) > 0:
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight < 0.0] = 0.0

    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight

class Training_all_eicu(object):
    def __init__(self, args, logger,Data,client_weights, train_loaders, valid_loaders, test_loaders, users_used=None):
        self.last_graph_matrix = None
        self.device = torch.device(
            'cuda:{}'.format(args.gpus[0]) if torch.cuda.is_available() and args.gpus != '-1' else 'cpu')
        self.args = args
        self.test_train = False
        self.last_net = None
        self.last_hnet = None
        self.net_last = None
        self.total_epoch = args.total_epoch
        self.epochs_per_valid = args.epochs_per_valid
        self.target_usr = args.target_usr
        self.emb_type = args.emb_type
        if users_used == None:
            self.users_used = [i for i in range(self.args.num_users)]
        else:
            self.users_used = users_used
        self.all_users = [i for i in range(self.args.num_users)]

        ################# DATA  #######################
        self.client_weights = client_weights
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.test_loaders = test_loaders
        self.dict_user_train = Data.dict_user_train
        self.num_batches = Data.num_batches
        total_data_points = sum([len(self.dict_user_train[r]) for r in users_used])
        self.traindata_cls_counts = Data.traindata_cls_counts
        self.fed_avg_freqs = [len(self.dict_user_train[r]) / total_data_points for r in users_used]
        if args.baseline_type == 'pfedgraph':
            self.fed_avg_freqs = {k: len(self.dict_user_train[k]) / total_data_points for k in users_used}

        ################# model  #######################
        if args.dataset=='eicu':
            self.hnet = TransformerModel(args=args, task=args.task, norm_layer='ln', user_used=self.users_used,
                                             device=self.device).to(self.device)

        elif args.dataset=='skin':
            # self.hnet =vgg(model_name="vgg16", num_classes=8, init_weights=True).to(self.device)
            self.hnet = SkinModel(args=args,emb_type=self.emb_type,user_used=self.users_used,device=self.device).to(self.device)


        self.hnet.to(self.device)
        if self.args.optim == 'sgd':
            self.optim = torch.optim.SGD(self.hnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optim == 'adam':
            self.optim = torch.optim.Adam(self.hnet.parameters(), lr=self.args.lr, weight_decay=args.weight_decay)

        self.schedulers = torch.optim.lr_scheduler.StepLR(self.optim, args.learning_rate_step, args.learning_rate_decay)

        if args.solver_type == "epo":
            self.solver = EPOSolver(len(self.users_used))
        elif args.solver_type == "linear":
            self.solver = LinearScalarizationSolver(len(self.users_used))
        self.logger = logger
        self.global_epoch = 0

    def train(self):

        if self.args.train_baseline:
            self.train_baseline()

        else:
            if self.args.sample_ray == True:
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                self.train_pt(0)

            else:
                results = {}
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                self.train_pt(0)
                for usr in self.users_used:
                    index = usr
                    criterion = self.task_losses(self.args.task)
                    valid_loss, valid_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, self.hnet,
                                                                                           self.valid_loaders[index],
                                                                                           criterion,
                                                                                           device=self.device, ray=None)
                    results[usr] = auroc_macro
                results = np.array(list(results.values()))

    def train_pt(self, target_usr):  # training the Pareto Front using training data
        start_epoch = 0
        if self.args.tensorboard:
            writer = SummaryWriter(self.args.tensorboard_dir)
        self.optim.param_groups[0]['lr'] = self.args.lr
        for iteration in range(start_epoch, self.args.total_hnet_epoch):
            self.hnet.train()

            losses = []
            accs = {}
            loss_items = {}

            if self.args.sample_ray:
                ray = torch.from_numpy(
                    np.random.dirichlet([self.args.beta for i in self.users_used], 1).astype(np.float32).flatten()).to(
                    self.device)
                ray = ray.view(1, -1)

                for usr_id in self.users_used:
                    index = usr_id
                    criterion = self.task_losses(self.args.task)
                    loss = self.train_naive(self.args, self.hnet, self.train_loaders[index], criterion, self.optim,
                                            device=self.device, scheduler=self.schedulers,ray = ray)

                    loss_items[str(self.args.target_usr)] = loss.item()
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()

                losses = torch.stack(losses)
                if self.args.tensorboard:
                    writer.add_scalar('hyperloss', torch.mean(losses), iteration)
                ray = self.hnet.input_ray.data
                ray = ray.squeeze(0)
                loss, alphas = self.solver(losses, ray,
                                           [p for n, p in self.hnet.named_parameters() if "local" not in n])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                target_usr = random.choice(self.users_used)
                index = target_usr
                criterion = self.task_losses(self.args.task)
                loss = self.train_naive(self.args, self.hnet, self.train_loaders[index], criterion, self.optim,
                                        device=self.device, scheduler=self.schedulers)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.hnet.input_ray.data[target_usr].add_(-self.hnet.input_ray.grad[target_usr] * self.args.lr_prefer)
                self.hnet.input_ray.data[target_usr] = torch.clamp(self.hnet.input_ray.data[target_usr],
                                                                   self.args.eps_prefer, 1)
                self.hnet.input_ray.data[target_usr] = self.hnet.input_ray.data[target_usr] / torch.sum(
                    self.hnet.input_ray.data[target_usr])

            if iteration % 20 == 0:
                for usr_id in self.users_used:
                    index = usr_id
                    valid_loss, valid_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, self.hnet,
                                                                                           self.valid_loaders[index],
                                                                                           criterion,
                                                                                           device=self.device)
                    self.logger.info('index{},auroc_macro:{},auprc_macro:{} '.format(index, auroc_macro,auprc_macro))

            self.logger.info('hyper iteration :{} losses: {} '.format(iteration, losses.data))

        if self.args.tensorboard:
            writer.close()
        self.args.tensorboard_dir = False

    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = torch.sum(adj, dim=1)
        rowsum = torch.flatten(rowsum)
        d_inv_sqrt = torch.pow(rowsum, -1.0)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.mm(d_mat_inv_sqrt, adj)

    def train_baseline(self):  ## training baselines (FedAve and Local)
        if self.args.baseline_type == "fedave":
            start_epoch = self.global_epoch
            criterion = self.task_losses(self.args.task)
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)
            for iteration in range(start_epoch, self.total_epoch):
                self.hnet.train()
                losses = []
                loss_items = {}
                for usr_id in self.users_used:
                    index = usr_id
                    loss = self.train_naive(self.args, self.hnet, self.train_loaders[index], criterion, self.optim,
                                            device=self.device, scheduler=self.schedulers)
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()

                    self.logger.info('index{},loss:{}, iteration:{} '.format(index, loss, iteration))

                loss = torch.mean(torch.stack(losses))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.args.tensorboard:
                    writer.add_scalar('fedaveloss', loss, iteration)

                self.global_epoch += 1
                if (self.global_epoch + 1) % self.epochs_per_valid == 0:
                    for usr_id in self.users_used:
                        index = usr_id
                        valid_loss, valid_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, self.hnet,
                                                                                               self.valid_loaders[
                                                                                                   index],
                                                                                               criterion,
                                                                                               device=self.device)
                        self.logger.info('index{},auroc_macro:{} '.format(index, auroc_macro))
            if self.args.tensorboard:
                writer.close()

        elif self.args.baseline_type == "local":
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)
            for iteration in range(0, self.total_epoch):
                self.hnet.train()
                loss_items = {}

                index = self.args.target_usr

                criterion = self.task_losses(self.args.task)

                loss = self.train_naive(self.args, self.hnet, self.train_loaders[index], criterion, self.optim,
                                        device=self.device, scheduler=self.schedulers)

                self.logger.info('index{},loss:{}, iteration:{} '.format(index, loss, iteration))

                loss_items[str(self.args.target_usr)] = loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.args.tensorboard:
                    writer.add_scalar('usr{}loss'.format(self.args.target_usr), loss, iteration)

                self.global_epoch += 1


                if (iteration + 1) % 50 == 0:
                    valid_loss, valid_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, self.hnet,
                                                                                           self.valid_loaders[index],
                                                                                           criterion,
                                                                                           device=self.device)
                    self.logger.info('index{},auroc_macro:{},auprc_macro:{}'.format(index, auroc_macro,auprc_macro))
            if self.args.tensorboard:
                writer.close()

        else:
            self.logger.info("error baseline type")
            exit()

    def task_losses(self, task):
        if task == "disch_24h" or task == "disch_48h":
            params = {'ignore_index': -1, 'reduction': 'none'}
            criterion = nn.CrossEntropyLoss(**params)
        elif task == 'Final Acuity Outcome':
            params = {'ignore_index': -1}
            criterion = nn.CrossEntropyLoss(**params)
        elif task == 'mort_24h' or task == 'mort_48h' or task == 'LOS':
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion

    def train_naive(self, args, f_model, train_dataloader, criterion, optimizer, device, scheduler=None, ray=None):
        f_model.train()

        total_loss = []

        if args.dataset =='skin':
            # for img,label in train_dataloader:
            img,label = next(iter(train_dataloader))
            optimizer.zero_grad()
            img,label = img.to(device),label.to(device).long()
            # if ray is not None:
            #     prediction,loss = f_model(img,label,0,input_ray = ray)
            # else:
            #     prediction,loss = f_model(img,label,0)
            if ray is not None:
                prediction = f_model(img,input_ray = ray)
            else:
                prediction = f_model(img)
            loss = criterion(prediction,label)
            total_loss.append(loss)
        else:
            for idx, batch in enumerate(train_dataloader):
                for k in (
                        'ts', 'statics',
                        'tasks_binary_multilabel'

                ):
                    if k in batch:
                        batch[k] = batch[k].to(device).float()
                        # batch[k] = batch[k].float().to(device)


                # optimizer.zero_grad()

                ts, statics = batch['ts'], batch['statics']

                if ray is not None:
                    logit = f_model(ts, statics, ray)
                else:
                    logit = f_model(ts, statics)

                if args.task in ['mort_24h', 'mort_48h']:
                    t = ['mort_24h', 'mort_48h'].index(args.task)
                    labels = batch['tasks_binary_multilabel'][:, t].unsqueeze(1)  # .to(device)

                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)

                    isnan = torch.isnan(labels).to(device)
                    labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
                    loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
                    loss = loss.mean()
                # loss.backward()
                # optimizer.step()
                # total_loss += loss.item()
                total_loss.append(loss)

            # torch.cuda.empty_cache()

        # if scheduler is not None:
        #     scheduler.step()
        mean_loss = torch.mean(torch.stack(total_loss))

        return mean_loss

    def test_naive(self, args, f_model, test_dataloader, criterion, device, ray=None):
        f_model.eval()

        binary_task = ['mort_24h', 'mort_48h']
        total_pred, total_label, total_score = [], [], []
        AUROC_macro, AUPRC_macro = 0.0, 0.0
        total_loss = []

        with torch.no_grad():
            if args.dataset == 'skin':
                labels = []
                label_list = []
                pred_scores = []
                # for img, label in test_dataloader:
                img, label = next(iter(test_dataloader))
                img, label = img.to(device), label.to(device).long()
                if ray is not None:
                    prediction = f_model(img, ray)
                else:
                    prediction = f_model(img)
                pred_scores.extend(prediction.cpu().detach().numpy())
                label_list.extend(((label.cpu()).numpy()).tolist())
                y_true = label.detach().cpu().numpy()
                labels.extend(y_true)
                loss = criterion(prediction, label)
                total_loss.append(loss)
                pred_scores = np.array(pred_scores)
                pred_scores = pred_scores[:, list(set(label_list))]
                predictions = np.argmax(pred_scores, axis=1)
                pred_scores = torch.nn.functional.softmax(torch.tensor(pred_scores), dim=-1).numpy()
                lb = preprocessing.LabelBinarizer()
                lb.fit(labels)
                AUROC_macro = 100.0 * average_precision_score(lb.transform(labels), lb.transform(predictions),
                                                              average='macro')
                AUPRC_macro = 100.0 * roc_auc_score(y_true=np.array(label_list), y_score=pred_scores, multi_class='ovr',
                                                    average='macro')
                test_loss = torch.mean(torch.stack(total_loss))

                return test_loss, 0, AUROC_macro, AUPRC_macro
            for idx, batch in enumerate(test_dataloader):
                for k in (
                        'ts', 'statics',
                        'tasks_binary_multilabel'
                ):
                    if k in batch:
                        batch[k] = batch[k].to(device).float()
                        # batch[k] = batch[k].float().to(device)



                ts, statics = batch['ts'], batch['statics']
                if ray is not None:
                    logit = f_model(ts, statics, ray)
                else:
                    logit = f_model(ts, statics)

                if args.task in binary_task:
                    t = binary_task.index(args.task)
                    labels = batch['tasks_binary_multilabel'][:, t].unsqueeze(1).to(device)
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)

                    isnan = torch.isnan(labels).to(device)

                    labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
                    loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
                    loss = loss.mean()

                    pred = logit.detach().cpu().numpy()
                    target = labels.detach().cpu().numpy()

                    total_pred.extend(pred)
                    total_label.extend(target)

                # total_loss += loss
                total_loss.append(loss)

        test_loss = torch.mean(torch.stack(total_loss))

        # torch.cuda.empty_cache()

        # test_loss = total_loss / len(test_dataloader)
        # test_loss = total_loss.mean()

        if args.task in binary_task:
            mask = np.isnan(total_label)
            post_label = np.array(total_label).astype(int)[~mask]
            post_pred = np.array(total_pred)[~mask]

            Accuracy = accuracy_score(post_label, np.where(post_pred < 0.5, 0, 1)) * 100
            AUROC_macro = roc_auc_score(post_label, post_pred, average='macro')
            AUPRC_macro = average_precision_score(post_label, post_pred, average='macro')
            return test_loss, Accuracy, AUROC_macro, AUPRC_macro



            return test_loss, Accuracy, auroc_macro, auprc_macro

    def valid_naive(self, args, f_model, test_dataloader, criterion, device, ray=None):
        f_model.eval()

        binary_task = ['mort_24h', 'mort_48h', 'LOS']
        total_pred, total_label, total_score = [], [], []
        AUROC_macro, AUPRC_macro = 0.0, 0.0
        total_loss = []

        if args.dataset == 'skin':
            labels = []
            label_list = []
            pred_scores = []
            # for img,label in test_dataloader:
            img, label = next(iter(test_dataloader))
            img,label = img.to(device),label.to(device).long()
            if ray is not None:
                prediction = f_model(img,ray)
            else:
                prediction = f_model(img)
            pred_scores.extend(prediction.cpu().detach().numpy())
            label_list.extend(((label.cpu()).numpy()).tolist())
            y_true = label.detach().cpu().numpy()
            labels.extend(y_true)
            loss = criterion(prediction,label)
            total_loss.append(loss)
            pred_scores = np.array(pred_scores)
            pred_scores = pred_scores[:, list(set(label_list))]
            predictions = np.argmax(pred_scores, axis=1)
            pred_scores = torch.nn.functional.softmax(torch.tensor(pred_scores), dim=-1).numpy()
            lb = preprocessing.LabelBinarizer()
            lb.fit(labels)
            AUROC_macro = 100.0 * average_precision_score(lb.transform(labels), lb.transform(predictions), average='macro')
            AUPRC_macro = 100.0 * roc_auc_score(y_true=np.array(label_list), y_score=pred_scores, multi_class='ovr',
                                            average='macro')
            test_loss = torch.mean(torch.stack(total_loss))

            return test_loss,0,AUROC_macro,AUPRC_macro

        for idx, batch in enumerate(test_dataloader):
            for k in (
                    'ts', 'statics',
                    'tasks_binary_multilabel'

            ):
                if k in batch:
                    batch[k] = batch[k].to(device).float()
                    # batch[k] = batch[k].float().to(device)

            ts, statics = batch['ts'], batch['statics']
            if ray is not None:
                logit = f_model(ts, statics, ray)
            else:
                logit = f_model(ts, statics)



            if args.task in binary_task:
                t = binary_task.index(args.task)
                labels = batch['tasks_binary_multilabel'][:, t].unsqueeze(1).to(device)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                isnan = torch.isnan(labels).to(device)

                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels).to(device)
                loss = torch.where(isnan, torch.zeros_like(logit).to(device), criterion(logit, labels_smoothed))
                loss = loss.mean()

                pred = logit.detach().cpu().numpy()
                target = labels.detach().cpu().numpy()

                total_pred.extend(pred)
                total_label.extend(target)


            # total_loss += loss
            total_loss.append(loss)

        test_loss = torch.mean(torch.stack(total_loss))

        # torch.cuda.empty_cache()

        # test_loss = total_loss / len(test_dataloader)
        # test_loss = total_loss.mean()

        if args.task in binary_task:
            mask = np.isnan(total_label)
            post_label = np.array(total_label).astype(int)[~mask]
            post_pred = np.array(total_pred)[~mask]

            Accuracy = accuracy_score(post_label, np.where(post_pred < 0.5, 0, 1)) * 100
            AUROC_macro = roc_auc_score(post_label, post_pred, average='macro')
            AUPRC_macro = average_precision_score(post_label, post_pred, average='macro')
            return test_loss, Accuracy, AUROC_macro, AUPRC_macro


    def test(self, usr):
        index = self.users_used.index(usr)
        criterion = self.task_losses(self.args.task)
        if self.args.baseline_type=='fedora' or  self.args.baseline_type=='pfedgraph' :
            if self.args.baseline_type=='fedora':
                net = self.net_last[index]
            else:
                net = self.net_last.get(index)
            test_loss, test_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, net,
                                                                                 self.test_loaders[usr], criterion,
                                                                                 device=self.device)
        else:

            test_loss, test_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, self.hnet,
                                                                                 self.test_loaders[index], criterion,
                                                                                 device=self.device)
        return test_loss, test_accuracy, auroc_macro, auprc_macro
    def benefit(self, usr_used, results=None,results1=None):
        if self.args.way_to_benefit=='graph':
            previous = self.args.baseline_type
            self.args.baseline_type='pfedgraph'
            self.train_baseline()
            benefit_matrix = self.last_graph_matrix
            self.args.baseline_type=previous
        else:
            benefit_matrix = []
            i = 0
            print('users')
            print(usr_used)
            for usr in usr_used:
                if (self.args.dataset == 'eicu' and self.args.num_users==5):
                    self.hnet.init_ray(usr, f=0)
                else:
                    self.hnet.init_ray(usr)
                self.args.target_usr = usr
                ray = self.train_ray(usr)
                print(ray)
                index = usr
                criterion = self.task_losses(self.args.task)
                valid_loss, valid_accuracy, auroc_macro, auprc_macro = self.test_naive(self.args, self.hnet,
                                                                                       self.valid_loaders[index],
                                                                                       criterion,
                                                                                       device=self.device, ray=ray)
                # accs, aucs, loss_dict = self.valid(ray=ray, target_usr=usr)
                benefit_matrix.append(ray.cpu().numpy()[0])
                if results is not None:
                    results[usr] = auroc_macro
                if results1 is not None:
                    results1[usr] = auprc_macro
                i += 1
        benefit_matrix = np.vstack(benefit_matrix)
        return benefit_matrix

    def train_ray(self, target_usr):  ## searching the optimal model on the PF using the validation data
        start_epoch = 0
        for iteration in range(start_epoch, self.args.total_ray_epoch):
            self.hnet.train()
            criterion = self.task_losses(self.args.task)
            index = target_usr
            valid_loss, valid_accuracy, auroc_macro, auprc_macro = self.valid_naive(self.args, self.hnet,
                                                                                    self.valid_loaders[index],
                                                                                    criterion,
                                                                                    device=self.device)
            self.optim.zero_grad()
            self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
            valid_loss.backward()
            self.logger.info('target_usr: {},  loss: {}'.format(target_usr,valid_loss))

            if self.args.sample_ray:
                self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * self.args.lr_prefer)
                self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, self.args.eps_prefer, 1)
                self.hnet.input_ray.data = self.hnet.input_ray.data / torch.sum(self.hnet.input_ray.data)
                input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]

            else:
                self.hnet.input_ray.data[target_usr].add_(-self.hnet.input_ray.grad[target_usr] * self.args.lr_prefer)
                self.hnet.input_ray.data[target_usr] = torch.clamp(self.hnet.input_ray.data[target_usr],
                                                                   self.args.eps_prefer, 1)
                self.hnet.input_ray.data[target_usr] = self.hnet.input_ray.data[target_usr] / torch.sum(
                    self.hnet.input_ray.data[target_usr])
            self.logger.info('ray iteration: {},  ray: {}'.format(iteration, self.hnet.input_ray.data))

        return self.hnet.input_ray.data

    def train_personal_hyper(self, user, users_used, type=1):
        criterion = self.task_losses(self.args.task)
        if (type):
            self.args.train_baseline = True
            self.args.baseline_type = "local"
            self.total_epoch = self.args.personal_init_epoch
            self.train_baseline()
            self.hnet.init_ray(users_used.index(user))
            self.args.baseline_type = 'fedcompetitors'
        else:
            self.args.train_baseline = True
            self.args.baseline_type = "local"
            self.total_epoch = self.args.personal_init_epoch
            self.train_baseline()
            self.hnet.init_ray(users_used.index(user))
            self.args.baseline_type = 'colequ'


        for iteration in range(self.args.personal_epoch):

            # ----------------------train hyper---------------------------
            self.hnet.train()
            losses = []
            for usr_id in self.users_used:
                index = usr_id
                criterion = self.task_losses(self.args.task)
                loss = self.train_naive(self.args, self.hnet, self.train_loaders[index], criterion, self.optim,
                                        device=self.device, scheduler=self.schedulers)
                losses.append(loss)

            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            input_ray_numpy = ray.data.cpu().numpy()
            loss, alphas = self.solver(losses, ray, [p for n, p in self.hnet.named_parameters() if "local" not in n])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.info("hyper iteration: {}, input_ray: {}.".format(iteration, input_ray_numpy))



            if (iteration % 10 == 0):
                test_loss, test_accuracy, auroc_macro, auprc_macro = self.test(user)

    def train_new(self, user, users_used):
        self.args.train_baseline = True
        self.args.baseline_type = "local"
        self.total_epoch = self.args.personal_init_epoch
        self.train_baseline()
        self.hnet.init_ray(users_used.index(user))
        self.args.baseline_type = 'fedegoists'
        criterion = self.task_losses(self.args.task)

        for iteration in range(self.args.personal_epoch):

            self.hnet.train()
            losses = []

            for usr_id in self.users_used:
                index = usr_id
                criterion = self.task_losses(self.args.task)
                loss = self.train_naive(self.args, self.hnet, self.train_loaders[index], criterion, self.optim,
                                        device=self.device, scheduler=self.schedulers)
                losses.append(loss)



            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            input_ray_numpy = ray.data.cpu().numpy()
            loss, alphas = self.solver(losses, ray, [p for n, p in self.hnet.named_parameters() if "local" not in n])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.info("hyper iteration: {}, input_ray: {}.".format(iteration, input_ray_numpy))




class Training_all(object):
    def __init__(self, args, logger, Data, users_used=None):
        self.last_net = None
        self.last_hnet = None
        self.net_last = None
        self.device = torch.device(
            'cuda:{}'.format(args.gpus[0]) if torch.cuda.is_available() and args.gpus != -1 else 'cpu')
        self.args = args
        self.test_train = False
        self.last_graph_matrix = None
        self.total_epoch = args.total_epoch
        self.epochs_per_valid = args.epochs_per_valid
        self.target_usr = args.target_usr
        self.users = []
        if users_used == None:
            self.users_used = [i for i in range(self.args.num_users)]
        else:
            self.users_used = users_used
        self.all_users = [i for i in range(self.args.num_users)]

        ################# DATA  #######################
        self.dataset_test = Data.dataset_test
        self.data_test = Data.dataset_test
        self.dataset_train = Data.dataset_train

        self.dict_user_train = Data.dict_user_train
        self.dict_user_test = Data.dict_user_test
        self.dict_user_valid = Data.dict_user_valid

        self.train_loaders = Data.train_loaders
        self.num_batches=Data.num_batches
        self.valid_loaders = Data.valid_loaders
        self.traindata_cls_counts = Data.traindata_cls_counts
        total_data_points = sum([len(self.dict_user_train[r]) for r in users_used])
        self.fed_avg_freqs = [len(self.dict_user_train[r]) / total_data_points for r in users_used]
        if args.baseline_type =='pfedgraph':
            self.fed_avg_freqs = {k: len(self.dict_user_train[k]) / total_data_points for k in users_used}


        ################# model  #######################
        if args.dataset=='eicu':
            self.hnet = TransformerModel1( task=args.task, norm_layer='ln',
                                             device=self.device ).to(self.device)

        elif args.dataset == "cifar10" or args.dataset=='cifar100':
            # if args.baseline_type == 'ours':
            if args.dataset=='cifar10':
                self.hnet = Hypernet(args=args, n_usrs=len(self.users_used), device=self.device, n_classes=args.n_classes,
                                     usr_used=self.users_used, n_hidden=args.n_hidden, spec_norm=args.spec_norm)
            elif args.dataset =='cifar100':
                self.hnet = CNNHyper100(args=args, n_nodes=len(self.users_used), device=self.device,out_dim=args.n_classes,
                                     usr_used=self.users_used, n_hidden=args.n_hidden, spec_norm=args.spec_norm)



        self.hnet.to(self.device)
        if self.args.optim == 'sgd':
            self.optim = torch.optim.SGD(self.hnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optim == 'adam':
            self.optim = torch.optim.Adam(self.hnet.parameters(), lr=self.args.lr, weight_decay=1e-5)

        if args.solver_type == "epo":
            self.solver = EPOSolver(len(self.users_used))
        elif args.solver_type == "linear":
            self.solver = LinearScalarizationSolver(len(self.users_used))
        self.logger = logger
        self.global_epoch = 0

    def change(self, target_user, users_used):
        self.target_usr = target_user
        self.users_used = users_used

    def all_args_save(self, args):
        with open(os.path.join(self.args.log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    def train_input(self, usr_id):
        try:
            _, (X, Y) = self.train_loaders[usr_id].__next__()
        except StopIteration:
            if self.args.local_bs == -1:
                t1 = time.time()
                self.train_loaders[usr_id] = enumerate(
                    FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[usr_id]),
                                   batch_size=len(self.dict_user_train[usr_id]), shuffle=True,
                                   num_workers=self.args.num_workers))
                t2 = time.time()
            else:
                self.train_loaders[usr_id] = enumerate(
                    FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[usr_id]),
                                   batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers))
            _, (X, Y) = self.train_loaders[usr_id].__next__()

        if self.args.dataset!='eicu':
            X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y

    def valid_input(self, usr_id):
        try:
            _, (X, Y) = self.valid_loaders[usr_id].__next__()
        except StopIteration:
            self.valid_loaders[usr_id] = enumerate(
                FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_valid[usr_id]),
                               batch_size=min(256, len(self.dict_user_valid[usr_id])), shuffle=True,
                               num_workers=self.args.num_workers))
            _, (X, Y) = self.valid_loaders[usr_id].__next__()
        if self.args.dataset != 'eicu':
            X = X.to(self.device)

        Y = Y.to(self.device)
        return X, Y

    def ray2users(self, ray):
        tmp_users_used = []
        tmp_ray = []
        for user_id, r in enumerate(ray):
            if r / ray[self.target_usr] >= 0.7:
                tmp_users_used.append(user_id)
                tmp_ray.append(r)
        return tmp_users_used, tmp_ray

    def acc_auc(self, prob, Y, is_training=True):
        if self.args.dataset == "adult" or self.args.dataset == "eicu":
            y_pred = prob.data >= 0.5
        elif self.args.dataset == "synthetic1" or self.args.dataset == "synthetic2" or self.args.dataset=='synthetic3':
            if is_training:
                return 0
            else:
                return 0, 0
        else:
            y_pred = prob.data.max(1)[1]
        users_acc = torch.mean((y_pred == Y).float()).item()

        if self.args.dataset == "eicu" and is_training:
            # users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
            return users_acc

        elif is_training and self.args.dataset != "eicu":
            return users_acc

        elif self.args.dataset == "eicu" and not is_training:
            users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
            return users_acc, users_auc
        else:
            return users_acc, 0

    def losses_r(self, l, ray):
        def mu(rl, normed=False):
            if len(np.where(rl < 0)[0]):
                raise ValueError(f"rl<0 \n rl={rl}")
                return None
            m = len(rl)
            l_hat = rl if normed else rl / rl.sum()
            eps = np.finfo(rl.dtype).eps
            l_hat = l_hat[l_hat > eps]
            return np.sum(l_hat * np.log(l_hat * m))

        m = len(l)
        rl = np.array(ray) * np.array(l)
        l_hat = rl / rl.sum()
        mu_rl = mu(l_hat, normed=True)
        return mu_rl

    def train_pt(self, target_usr):  # training the Pareto Front using training data
        start_epoch = 0
        if self.args.tensorboard:
            writer = SummaryWriter(self.args.tensorboard_dir)
        for iteration in range(start_epoch, self.args.total_hnet_epoch):
            self.hnet.train()
            losses = []
            accs = {}
            loss_items = {}

            if self.args.sample_ray:
                ray = torch.from_numpy(
                    np.random.dirichlet([self.args.beta for i in self.users_used], 1).astype(np.float32).flatten()).to(
                    self.device)
                ray = ray.view(1, -1)
                for usr_id in self.users_used:
                    X, Y = self.train_input(usr_id)
                    pred, loss = self.hnet(X, Y, usr_id, ray)
                    acc = self.acc_auc(pred, Y)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()
                losses = torch.stack(losses)
                if self.args.tensorboard:
                    writer.add_scalar('hyperloss', torch.mean(losses), iteration)
                ray = self.hnet.input_ray.data
                ray = ray.squeeze(0)
                loss, alphas = self.solver(losses, ray,
                                           [p for n, p in self.hnet.named_parameters() if "local" not in n])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                target_usr = random.choice(self.users_used)
                X, Y = self.train_input(target_usr)
                pred, loss = self.hnet(X, Y, target_usr)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.hnet.input_ray.data[target_usr].add_(-self.hnet.input_ray.grad[target_usr] * self.args.lr_prefer)
                self.hnet.input_ray.data[target_usr] = torch.clamp(self.hnet.input_ray.data[target_usr],
                                                                   self.args.eps_prefer, 1)
                self.hnet.input_ray.data[target_usr] = self.hnet.input_ray.data[target_usr] / torch.sum(
                    self.hnet.input_ray.data[target_usr])

            if iteration % self.args.epochs_per_valid == 0:
                _, _, loss_dict = self.valid()

                self.logger.info('hyper iteration :{} losses: {} '.format(iteration, losses.data))

        if self.args.tensorboard:
            writer.close()
        self.args.tensorboard_dir = False

    def train_ray(self, target_usr):  ## searching the optimal model on the PF using the validation data

        start_epoch = 0
        for iteration in range(start_epoch, self.args.total_ray_epoch):
            self.hnet.train()
            X, Y = self.valid_input(target_usr)
            pred, loss = self.hnet(X, Y, target_usr)
            acc = self.acc_auc(pred, Y)
            self.optim.zero_grad()
            self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
            loss.backward()

            if self.args.sample_ray:
                self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * self.args.lr_prefer)
                self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, self.args.eps_prefer, 1)
                self.hnet.input_ray.data = self.hnet.input_ray.data / torch.sum(self.hnet.input_ray.data)
                input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]

            else:
                self.hnet.input_ray.data[target_usr].add_(-self.hnet.input_ray.grad[target_usr] * self.args.lr_prefer)
                self.hnet.input_ray.data[target_usr] = torch.clamp(self.hnet.input_ray.data[target_usr],
                                                                   self.args.eps_prefer, 1)
                self.hnet.input_ray.data[target_usr] = self.hnet.input_ray.data[target_usr] / torch.sum(
                    self.hnet.input_ray.data[target_usr])
            self.logger.info('ray iteration: {},  ray: {}'.format(iteration, self.hnet.input_ray.data))

        return self.hnet.input_ray.data

    def train_baseline(self):  ## training baselines (FedAve and Local)
        if self.args.baseline_type == "fedave":
            start_epoch = self.global_epoch
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)
            for iteration in range(start_epoch, self.total_epoch):
                self.hnet.train()

                losses = []
                accs = {}
                loss_items = {}

                for usr_id in self.users_used:
                    X, Y = self.train_input(usr_id)
                    pred, loss = self.hnet(X, Y, usr_id)
                    acc = self.acc_auc(pred, Y)
                    accs[str(usr_id)] = acc
                    losses.append(loss)
                    loss_items[str(usr_id)] = loss.item()

                loss = torch.mean(torch.stack(losses))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.args.tensorboard:
                    writer.add_scalar('fedaveloss', loss, iteration)

                self.logger.info('iteration: {}'.format(iteration))
                self.global_epoch += 1
                if (self.global_epoch + 1) % self.epochs_per_valid == 0:
                    self.valid()

            if self.args.tensorboard:
                writer.close()

            start_epoch = self.global_epoch
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)

        elif self.args.baseline_type == "local":
            if self.args.tensorboard:
                writer = SummaryWriter(self.args.tensorboard_dir)
            for iteration in range(0, self.total_epoch):
                self.hnet.train()
                loss_items = {}

                X, Y = self.train_input(self.args.target_usr)

                pred, loss = self.hnet(X, Y, self.args.target_usr)
                acc = self.acc_auc(pred, Y)
                loss_items[str(self.args.target_usr)] = loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.args.tensorboard:
                    writer.add_scalar('usr{}loss'.format(self.args.target_usr), loss, iteration)

                self.global_epoch += 1
                if iteration % 1 == 0:
                    if self.args.personal_init_epoch != 0:
                        self.logger.info('init iteration:{} '.format(iteration))


                    else:
                        self.logger.info('local iteration:{} '.format(iteration))

                if (self.global_epoch + 1) % self.epochs_per_valid == 0:
                    self.valid()
            if self.args.tensorboard:
                writer.close()

        else:
            self.logger.info("error baseline type")
            exit()

    def train(self):

        if self.args.train_baseline:
            self.train_baseline()

        else:
            if self.args.sample_ray == True:
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                self.train_pt(0)

            else:
                results = {}
                self.total_epoch = self.args.total_epoch
                self.epochs_per_valid = self.args.epochs_per_valid
                self.train_pt(0)
                for usr in self.users_used:
                    accs, aucs, _ = self.valid(ray=None, target_usr=usr)
                    self.logger.info('usr :{},  weight: {}'.format(usr, self.hnet.input_ray.data[usr]))

                    if self.args.dataset == 'eicu':
                        results[usr] = aucs[str(usr)]
                    else:
                        results[usr] = accs[str(usr)]
                results = np.array(list(results.values()))

    def train_personal_hyper(self, user, users_used, type=1):
        if (type):
            self.args.train_baseline = True
            self.args.baseline_type = "local"
            self.total_epoch = self.args.personal_init_epoch
            if self.args.dataset=='cifar10' or self.args.dataset=='cifar100':
                self.train_baseline()
            self.hnet.init_ray(users_used.index(user))
            self.args.baseline_type = 'fedcompetitors'
        # else:
        #     self.args.train_baseline = True
        #     self.args.baseline_type = "local"
        #     self.total_epoch = self.args.personal_init_epoch
        #     self.train_baseline()
        #     self.hnet.init_ray(users_used.index(user))
        #     self.args.baseline_type = 'colequ'


        for iteration in range(self.args.personal_epoch):

            # ----------------------train hyper---------------------------
            self.hnet.train()
            losses = []
            accs = {}
            loss_items = {}

            for usr_id in self.users_used:
                X, Y = self.train_input(usr_id)
                pred, loss = self.hnet(X, Y, usr_id)
                acc = self.acc_auc(pred, Y)
                accs[str(usr_id)] = acc
                losses.append(loss)
                loss_items[str(usr_id)] = loss.item()

            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            input_ray_numpy = ray.data.cpu().numpy()
            loss, alphas = self.solver(losses, ray, [p for n, p in self.hnet.named_parameters() if "local" not in n])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.info("hyper iteration: {}, input_ray: {}.".format(iteration, input_ray_numpy))

            # ------------------------------train ray-----------------------------
            self.hnet.train()
            X, Y = self.valid_input(user)
            if self.args.dataset !='eicu':
                X = X.to(self.device)

            Y = Y.to(self.device)
            pred, loss = self.hnet(X, Y, user)
            acc = self.acc_auc(pred, Y)
            self.optim.zero_grad()
            self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
            loss.backward()

            self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * self.args.lr_prefer)
            self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, self.args.eps_prefer, 1)
            self.hnet.input_ray.data = self.hnet.input_ray.data / torch.sum(self.hnet.input_ray.data)
            input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]

            self.logger.info('ray iteration: {},  ray: {}'.format(iteration, self.hnet.input_ray.data))

            if (iteration % 10 == 0):
                acc, auc, loss = self.personal_test(user)

    def train_new(self, user, users_used):
        self.args.train_baseline = True
        self.args.baseline_type = "local"
        self.total_epoch = self.args.personal_init_epoch
        if self.args.dataset == 'cifar10':
            self.train_baseline()
        self.hnet.init_ray(users_used.index(user))
        self.args.baseline_type = 'fedegoists'

        for iteration in range(self.args.personal_epoch):

            self.hnet.train()
            losses = []
            accs = {}
            loss_items = {}
            self.hnet.train()

            for usr_id in self.users_used:
                X, Y = self.train_input(usr_id)
                pred, loss = self.hnet(X, Y, usr_id)
                acc = self.acc_auc(pred, Y)
                accs[str(usr_id)] = acc
                losses.append(loss)
                loss_items[str(usr_id)] = loss.item()

            losses = torch.stack(losses)
            ray = self.hnet.input_ray.data
            ray = ray.squeeze(0)
            input_ray_numpy = ray.data.cpu().numpy()
            loss, alphas = self.solver(losses, ray, [p for n, p in self.hnet.named_parameters() if "local" not in n])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.logger.info("hyper iteration: {}, input_ray: {}.".format(iteration, input_ray_numpy))

            X, Y = self.valid_input(user)
            pred, loss = self.hnet(X, Y, user)
            acc = self.acc_auc(pred, Y)
            self.optim.zero_grad()
            self.hnet.input_ray.grad = torch.zeros_like(self.hnet.input_ray.data)
            loss.backward()
            self.hnet.input_ray.data.add_(-self.hnet.input_ray.grad * self.args.lr_prefer)
            self.hnet.input_ray.data = torch.clamp(self.hnet.input_ray.data, self.args.eps_prefer, 1)
            self.hnet.input_ray.data = self.hnet.input_ray.data / torch.sum(self.hnet.input_ray.data)
            input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]

            self.logger.info('ray iteration: {},  ray: {}'.format(iteration, self.hnet.input_ray.data))

            if (iteration % 10 == 0):
                acc, auc, loss = self.personal_test(user)

    def benefit(self, usr_used, results=None):
        if self.args.way_to_benefit == 'graph':
            previous = self.args.baseline_type
            self.args.baseline_type = 'pfedgraph'
            self.args.train_baseline = True
            self.train_baseline()
            benefit_matrix = self.last_graph_matrix
            self.args.train_baseline = False
            self.args.baseline_type = previous
        else:
            benefit_matrix = []
            i = 0
            print('users')
            print(usr_used)
            for usr in usr_used:
                print(self.args.train_pt)
                self.hnet.init_ray(usr, 0)
                self.args.target_usr = usr
                ray = self.train_ray(usr)
                print(ray)
                accs, aucs, loss_dict = self.valid(ray=ray, target_usr=usr)

                benefit_matrix.append(ray.cpu().numpy()[0])
                if results is not None:
                    if self.args.dataset == 'eicu':
                        results[usr] = aucs[str(usr)]
                    else:
                        results[usr] = accs[str(usr)]
                i += 1

        benefit_matrix = np.vstack(benefit_matrix)

        return benefit_matrix

    def test_input(self, data_loader):
        _, (X, Y) = data_loader.__next__()
        if self.args.dataset!='eicu':
            X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable

    def losses_r(self, l, ray):
        def mu(rl, normed=False):
            if len(np.where(rl < 0)[0]):
                raise ValueError(f"rl<0 \n rl={rl}")
                return None
            m = len(rl)
            l_hat = rl if normed else rl / rl.sum()
            eps = np.finfo(rl.dtype).eps
            l_hat = l_hat[l_hat > eps]
            return np.sum(l_hat * np.log(l_hat * m))

        m = len(l)
        rl = np.array(ray) * np.array(l)
        l_hat = rl / rl.sum()
        mu_rl = mu(l_hat, normed=True)
        return mu_rl

    def valid_baseline(self, model="baseline", target_usr=0, load=False, ckptname="last", train_data=False):
        with torch.no_grad():
            if train_data:
                data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[idxs]),
                                                         batch_size=len(self.dict_user_train[idxs]), shuffle=False,
                                                         drop_last=False))
                                for idxs in range(self.args.num_users)]
            else:
                data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]),
                                                         batch_size=len(self.dict_user_test[idxs]), shuffle=False,
                                                         drop_last=False))
                                for idxs in range(self.args.num_users)]
            if load:
                self.load_hnet(ckptname)

            accs = {}
            loss_dict = {}
            loss_list = []
            aucs = {}

            if self.args.baseline_type == 'pfedgraph' or self.args.baseline_type == 'pfedme':
                for id,net in self.net_last.items():
                    usr_id = self.users_used[id]
                    X, Y = self.test_input(data_loaders[usr_id])
                    if self.args.dataset == 'eicu':
                        criterion = nn.BCELoss()
                        pred = net(X)
                        loss = criterion(pred, Y.float())
                    else:
                        pred, loss = net(X, Y, usr_id)
                    # pred, loss = net(X, Y, usr_id)
                    acc, auc = self.acc_auc(pred, Y, is_training=False)
                    accs[str(usr_id)] = acc
                    loss_dict[str(usr_id)] = loss.item()
                    loss_list.append(loss.item())
                    aucs[str(usr_id)] = auc
            elif self.args.baseline_type == 'pfedhn':
                criteria = torch.nn.CrossEntropyLoss()
                for usr_id in self.users_used:
                    X, Y = self.test_input(data_loaders[usr_id])
                    client_id = self.users_used.index(usr_id)
                    weights = self.last_hnet(torch.tensor([client_id], dtype=torch.long).to(self.device))
                    self.last_net.load_state_dict(weights)
                    pred = self.last_net(X)
                    loss = criteria(pred, Y)
                    acc, auc = self.acc_auc(pred, Y, is_training=False)
                    accs[str(usr_id)] = acc
                    loss_dict[str(usr_id)] = loss.item()
                    loss_list.append(loss.item())
                    aucs[str(usr_id)] = auc
            else:
                for usr_id in self.users_used:
                    X, Y = self.test_input(data_loaders[usr_id])
                    pred, loss = self.hnet(X, Y,usr_id)
                    acc, auc = self.acc_auc(pred, Y, is_training=False)
                    accs[str(usr_id)] = acc
                    loss_dict[str(usr_id)] = loss.item()
                    loss_list.append(loss.item())
                    aucs[str(usr_id)] = auc
            self.logger.info('accs{} aucs{} loss_dict{}'.format(accs,aucs,loss_dict))
            self.logger.info('mta{}'.format(np.mean(np.mean(list(accs.values())))))

            return accs, aucs, loss_dict

    def valid(self, model="hnet", ray=None, target_usr=0, load=False, ckptname="last", train_data=False):
        target_usr = self.target_usr
        self.hnet.eval()
        if self.args.train_baseline:
            accs, aucs, loss_dict = self.valid_baseline()
            return accs, aucs, loss_dict
        else:
            with torch.no_grad():
                if train_data:
                    data_loaders = [enumerate(
                        FastDataLoader(DatasetSplit(self.dataset_train, self.dict_user_train[idxs]),
                                       batch_size=len(self.dict_user_train[idxs]), shuffle=False, drop_last=False))
                        for idxs in range(self.args.num_users)]
                else:
                    data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]),
                                                             batch_size=len(self.dict_user_test[idxs]), shuffle=True,
                                                             drop_last=False))
                                    for idxs in range(self.args.num_users)]
                if load:
                    self.load_hnet(ckptname)

                accs = {}
                loss_dict = {}
                loss_list = []
                aucs = {}

                for usr_id in self.users_used:
                    if self.test_train:
                        X, Y = self.train_input(usr_id)
                    else:
                        X, Y = self.test_input(data_loaders[usr_id])
                    pred, loss = self.hnet(X, Y, usr_id, ray)
                    acc, auc = self.acc_auc(pred, Y, is_training=False)
                    accs[str(usr_id)] = acc
                    loss_dict[str(usr_id)] = loss.item()
                    loss_list.append(loss.item())
                    aucs[str(usr_id)] = auc

                input_ray_numpy = self.hnet.input_ray.data.cpu().numpy()[0]
                self.logger.info('accs{} aucs{} loss_dict{}'.format(accs, aucs, loss_dict))

                return accs, aucs, loss_dict

    def personal_test(self, user):
        with torch.no_grad():
            data_loaders = [enumerate(FastDataLoader(DatasetSplit(self.dataset_test, self.dict_user_test[idxs]),
                                                     batch_size=len(self.dict_user_test[idxs]), shuffle=False,
                                                     drop_last=False))
                            for idxs in range(self.args.num_users)]
            X, Y = self.test_input(data_loaders[user])
            pred, loss = self.hnet(X, Y, user)
            acc, auc = self.acc_auc(pred, Y, is_training=False)
        return acc, auc, loss.item()

    def save_hnet(self, ckptname=None):
        states = {'epoch': self.global_epoch,
                  'model': self.hnet.state_dict(),
                  'optim': self.optim.state_dict(),
                  'input_ray': self.hnet.input_ray.data}
        if ckptname == None:
            ckptname = str(self.global_epoch)
        os.makedirs(self.args.hnet_model_dir, exist_ok=True)
        filepath = os.path.join(self.args.hnet_model_dir, str(ckptname))
        print(filepath)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> hnet saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))

    def load_hnet(self, ckptname='last'):
        if ckptname == 'last':
            ckpts = os.listdir(self.args.hnet_model_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.args.hnet_model_dir, str(ckptname))
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.global_epoch = checkpoint['epoch']
            self.hnet.load_state_dict(checkpoint['model'])
            # self.hnet.input_ray.data = checkpoint["input_ray"].data.view(1, -1)
            self.optim.load_state_dict(checkpoint['optim'])
            self.logger.info("=> hnet loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))

