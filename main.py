# main.py
import copy
import random

import networkx as nx
import numpy as np
import torch

from utils.utils_data import get_data, get_dataset, get_dataloader, get_federated_dataset, get_dataset1
from utils.utils_func import construct_log, get_random_dir_name, setup_seed
from hyper_model.benefit import Training_all, Training_all_eicu
from hyper_model.graph import  Generate_competetive_graph, generate_coalition_group, Node, Tarjan, \
   generate_group
from hyper_model.data import data
import os

import argparse
import time

parser = argparse.ArgumentParser()
# federated arguments
parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user (each user has the num of classes)")
parser.add_argument('--target_usr', type=int, default=2, help="target usr id")

# training arguments

parser.add_argument('--total_hnet_epoch', type=int, default=500, help="hnet update steps for train_pt")
parser.add_argument('--total_ray_epoch', type=int, default=500, help="hnet update innner steps")
parser.add_argument('--total_epoch', type=int, default=500, help="update steps")
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--local_bs', type=int, default=256, help="local batch size: B")
parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
parser.add_argument('--server_learning_rate', type=float, default=1)
parser.add_argument('--lr_prefer', type=float, default=0.01, help="learning rate for preference vector")
parser.add_argument('--alpha', type=float, default=0.2, help="alpha for sampling the preference vector")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
parser.add_argument('--epochs_per_valid', type=int, default=10, help="rounds of valid")
parser.add_argument('--num_workers', type=int, default=0, help="the number of workers for the dataloader.")
parser.add_argument('--eps_prefer', type=float, default=0.001, help="learning rate for preference vector")
parser.add_argument('--sigma', type=float, default=0.1, help="learning rate for preference vector")
parser.add_argument('--std', type=float, default=0.01, help="learning rate for preference vector")
parser.add_argument('--trainN', type=int, default=2000, help="the number of generated train samples .")
parser.add_argument('--testN', type=int, default=2000, help="the number of generated test samples.")
parser.add_argument('--solver_type', type=str, default="linear", help="the type of solving the model,linear,epo")
parser.add_argument('--sample_ray', action='store_true', default=True,
                    help='whether sampling alpha for learning Pareto Front')
parser.add_argument('--train_baseline', action='store_true', default=False,
                    help='whether train baseline for eicu dataset')
parser.add_argument('--baseline_type', type=str, default="ours", help="the type of training")
parser.add_argument('--optim', type=str, default='adam', help="optimizer setting")
# personal model training parameters

parser.add_argument('--train_pt', action='store_true', default=True, help='train the Pareto Front')
parser.add_argument('--compete_ratio', type=float, default=0.2, help='compete ratio')
parser.add_argument('--personal_init_epoch', type=int, default=500, help='personal init epoch')
parser.add_argument('--personal_epoch', type=int, default=500, help="personal model training epoch")

# model structure
parser.add_argument('--n_classes', type=int, default=10, help="the number of classes.")
parser.add_argument('--entropy_weight', type=float, default=0.0, help="the number of classes.")
parser.add_argument('--n_hidden', type=int, default=2, help="hidden layer for the hypernet.")
parser.add_argument('--embedding_dim', type=int, default=-1,
                    help="embedding dim for eicu embedding the categorical features")
parser.add_argument('--input_dim', type=int, default=20, help="input dim (generate dim) for the hypernet.")
parser.add_argument('--output_dim', type=int, default=2, help="hidden layer for the hypernet.")
parser.add_argument('--hidden_dim', type=int, default=100, help="hidden dim for the hypernet.")
parser.add_argument('--spec_norm', action='store_true', help='whether using spectral norm not')
parser.add_argument('--partition', default='kdd', help='partition type')
parser.add_argument('--epsilon', type=float, default=0.1, help="selected hosptial id in eicu dataset")
parser.add_argument('--mu', type=float, default=0.01, help="selected hosptial id in eicu dataset")
parser.add_argument('--beta', type = float,default=0.1,
                    help="parameter for dirichlet distribution, the larger, the similar the variables are ")
parser.add_argument('--dirbeta', type = float,default=0.5,
                    help="parameter for dirichlet distribution, the larger, the similar the variables are ")
parser.add_argument('--selected_hospital', default=False, help="selected hosptial id in eicu dataset")
parser.add_argument('--runs', type=int, default=0, help="run times")
# learning setup arguments
parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--auto_deploy', action='store_true', help='whether auto deploy not')
# devices
parser.add_argument('--gpus', type=str, default="0", help='gpus for training')
# dataset/log/outputs/ dir
parser.add_argument('--dataset', type=str, default='synthetic2', help="name of dataset")
parser.add_argument('--data_root', type=str, default='data', help="name of dataset")
parser.add_argument('--outputs_root', type=str, default='outputs', help="name of dataset")
parser.add_argument('--target_dir', type=str, default='', help=" dir name of for saving all generating data")
parser.add_argument('--tensorboard', default=False, help="whether tensorboard is used")
parser.add_argument('--tensorboard_dir', type=str, default='', help=" dir name of tensorboard logs")
parser.add_argument('--load_dir', type=str, default='', help="dir name of tensorboard logs")

# eicu argument
parser.add_argument('--dropout', type=float, default=0.3, help="Dropout")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
parser.add_argument('--weight_decay', type=float, default=0.00001, help="Weight decay")
parser.add_argument('--learning_rate_step', type=float, default=33, help="Learning rate step")  # 10
parser.add_argument('--learning_rate_decay', type=float, default=5, help="Learning rate step")  # 0.5
parser.add_argument('--epochs', type=int, default=50, help="Epochs")
parser.add_argument("--K", type=int, default=5, help="Computation steps")
parser.add_argument('--model_type', type=str, default='transformer_ln', help="transformer_ln")
parser.add_argument('--task', type=str, default='mort_48h',
                    help="mort_24h|mort_48h")
parser.add_argument('--way_to_benefit', type=str, default='hyper',
                    help="hyper|graph")

parser.add_argument('--data_path', type=str, default='data_storage', help="data path")
parser.add_argument('--save_path', type=str, default='eicu_checkpoint', help="save path")
tp = lambda x: list(map(str, x.split('.')))
parser.add_argument('--hospital_id', type=tp, default="300.148.365.413.183.154.301.184.152.391", help="hospital id list")
#300.148.365.413.183
#300.148.365.413.183.154.301.184.152.391
#skin argument
tp = lambda x: list(map(str, x.split('.')))
#barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt
parser.add_argument('--clients', type=tp, default="barcelona.rosendahl.vienna.PAD_UFES_20.Derm7pt", help="client list")
parser.add_argument('--emb_type', type=str, default='mobilenet', help="efficientnet|efficientnet_gn")

parser.add_argument('--measure_difference', type=str, default='kl',
                    help='how to measure difference. e.g. only_iid, cosine')
parser.add_argument('--disco_a', type=float, default=0.5, help='under sub mode, n_k-disco_a*d_k+disco_b')
parser.add_argument('--disco_b', type=float, default=0.1)

parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
parser.add_argument('--lam', type=float, default=0.01, help="Hyper-parameter in the objective")
parser.add_argument("--local_epochs", type=int , default=1)
parser.add_argument("--personal_learning_rate", type=float, default=0.01,
                    help="Persionalized learning rate to caculate theta aproximately using K steps")
parser.add_argument("--inner_lr", type=float, default=5e-3, help="learning rate for inner optimizer")
parser.add_argument("--embed_lr", type=float, default=None, help="embedding learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
parser.add_argument("--inner_wd", type=float, default=5e-5, help="inner weight decay")
parser.add_argument("--n_kernels", type=int, default=16, help="number of kernels for cnn model")
parser.add_argument("--eval_every", type=int, default=30, help="eval every X selected epochs")


parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
parser.add_argument('--attack_type', type=str, default="inv_grad")
parser.add_argument('--attack_ratio', type=float, default=0.0)
parser.add_argument('--difference_measure', type=str, default='all', help='How to measure the model difference')
args = parser.parse_args()

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
fname = "./runs/" + now + args.dataset

args.tensorboard_dir = True
args.auto_deploy = True
args.load_dir = args.target_dir

def SCC(args,group,logger=None,Data=None, client_weights=None, train_loaders=None, valid_loaders=None, test_loaders=None):
    S = []  # Collaboration Strategy
    C = group
    ajadcent_Coalition_Matrix = np.zeros([args.num_users, args.num_users])
    while len(C):  # C is not an empty set
        # print("C{}".format(C))
        if args.dataset == 'eicu':
            model = Training_all_eicu(args, logger, Data, client_weights, train_loaders, valid_loaders, test_loaders,
                                      users_used=C)
            model.train()
            Benefit_Matrix = model.benefit(C)
        else:
            model = Training_all(args, logger, Data, users_used=C)
            model.train()  # Calculate the weight preference for each client
            Benefit_Matrix = model.benefit(C)
        logger.info('Benefit_Matrix{}'.format(Benefit_Matrix))

        epsilon = args.epsilon
        Benefit_Matrix = np.vstack(Benefit_Matrix.T)
        Benefit_Matrix = Benefit_Matrix * (Benefit_Matrix > epsilon)  # Set the value below the epsilon in the benefit matrix to 0
        ajadcent_Benefit_Matrix = np.zeros([args.num_users, args.num_users])  # the Adjacent Matrix of Benefit Graph

        print(len(Benefit_Matrix))
        for i in range(len(Benefit_Matrix)):
            for j in range(len(Benefit_Matrix)):
                if (Benefit_Matrix[i, j] == 0):
                    ajadcent_Benefit_Matrix[C[i], C[j]] = 0
                else:
                    ajadcent_Benefit_Matrix[C[i], C[j]] = 1
        for i in range(args.num_users):
            ajadcent_Benefit_Matrix[i, i] = 0
        print(ajadcent_Benefit_Matrix)

        # ------------------------------The Constrcution of Benefit Graph--------------------------------#
        nodes = {
            i: Node(id=i,
                    parents=[j for j in range(len(ajadcent_Benefit_Matrix)) if ajadcent_Benefit_Matrix[j][i]],
                    descendants=[j for j in range(len(ajadcent_Benefit_Matrix)) if
                                 ajadcent_Benefit_Matrix[i][j]])
            for i in range(len(ajadcent_Benefit_Matrix))}
        # algorithm tarjan find SCC
        tarjan = Tarjan(nodes=nodes)
        C_tmp = C
        for scc in tarjan.list_scc:  # each SCC
            A = [True for i in scc if i not in C]
            if (len(A) == 1 and A[0] == True):
                continue
            # determine whether it is a stable coalition
            f = 1
            for i in scc:
                for j in C_tmp:
                    if ajadcent_Benefit_Matrix[j][i] == 1 and j not in scc:
                        f = 0
            if (f):  # If it is a stable coalition
                S.append(scc)
                for i in scc:
                    for j in scc:
                        ajadcent_Coalition_Matrix[i, j] = ajadcent_Benefit_Matrix[i, j]
                C = [elem for elem in C if elem not in scc]
    return S,ajadcent_Coalition_Matrix

def Merge(X,Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger):
    tmp = []
    l = len(Z_b)+1

    for i in X:
        l = l-1
        tmp.append(new_group[i])
    new_group = [new_group[i] for i in range(len(new_group)) if i not in X]

    new_group.append([item for sublist in tmp for item in sublist])

    Z_b_new = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            if j == i:
                continue
            f = 0
            for m in new_group[i]:
                for n in new_group[j]:
                    if aja_Bene[m][n] == 1:
                        f = 1
            if f:
                Z_b_new[i][j] = 1

    Z_c_new = np.zeros([l, l])
    for i in range(l):
        for j in range(i + 1, l):
            f = 0
            for m in new_group[i]:
                for n in new_group[j]:
                    if Compet_Matrix[m][n] == 1:
                        f = 1
            if f:
                Z_c_new[i][j] = 1
                Z_c_new[j][i] = 1

    return Z_b_new,Z_c_new,new_group


def MergeCycle(Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger):
    Y = len(new_group)
    flag = np.zeros(Y)
    print(flag)
    while (1):
        nodes = []
        for i in range(Y):
            if len(new_group[i]) == 1 and flag[i] == 0:
                nodes.append(i)

        if len(nodes) == 0:
            break
        else:
            node = random.choice(nodes)
            print(node)
            G_z = nx.from_numpy_matrix(Z_b, create_using=nx.DiGraph)
            all_cycles = list(nx.simple_cycles(G_z))
            cycle_containing_node = [cycle for cycle in all_cycles if node in cycle]  # cycle in zb contain node
            if len(cycle_containing_node) == 0:
                flag[node] = 1
                continue

            cycle_containing_node.sort(key=len, reverse=True)
            cnt = 0
            tmp_cycle = []
            for cycle in cycle_containing_node:
                f = 1
                print(cycle)
                for m in cycle_containing_node[0]:
                    for n in cycle_containing_node[0]:
                        if Z_c[m, n] == 1 and m != n:
                            f = 0
                if (f):
                    tmp_cycle.append(cycle)
                    cnt = cnt + 1
            if (cnt):
                Z_b, Z_c, new_group = Merge(tmp_cycle[0], Z_b, Z_c, new_group,aja_Bene,Compet_Matrix,logger)
                Y = len(new_group)
                flag = np.zeros(Y)
            else:
                flag[node] = 1
                Y = len(new_group)
        Y = len(new_group)
    return Z_b,Z_c,new_group

def MergePath(Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger):
    Y = len(new_group)
    flag1 = np.zeros(Y)
    while (1):
        nodes = []
        for i in range(Y):
            if len(new_group[i]) == 1 and flag1[i] == 0:
                nodes.append(i)
        if len(nodes) == 0:
            break
        else:
            node = random.choice(nodes)
            G_z = nx.from_numpy_matrix(Z_b, create_using=nx.DiGraph)
            all_cycles = list(nx.simple_cycles(G_z))
            cycle_containing_node = [cycle for cycle in all_cycles if node in cycle]
            f = 0
            for v_s in range(len(Z_b)):
                for v_t in range(len(Z_b)):
                    if v_t != v_s and nx.has_path(G_z, source=v_s, target=v_t):
                        if (v_s != node and v_t != node and len(new_group[v_s]) >= 2 and len(new_group[v_t]) >= 2) or (
                                v_s == node and len(new_group[v_t]) >= 2) or (v_t == node and len(new_group[v_s]) >= 2):
                            paths = list(nx.all_simple_paths(G_z, v_s, v_t))
                            simpaths = [sub for sub in paths if sub not in cycle_containing_node]
                            havepaths = [path for path in simpaths if node in path]
                            if len(havepaths) == 0:
                                continue
                            havepaths.sort(key=len, reverse=True)
                            print(havepaths)
                            cnt = 0
                            tmp_path = []
                            for path in havepaths:
                                f = 1
                                print(path)
                                for m in path:
                                    for n in path:
                                        if Z_c[m, n] == 1 and m != n:
                                            f = 0
                                if (f):
                                    tmp_path.append(path)
                                    cnt = cnt + 1

                            if (cnt):
                                Z_b, Z_c, new_group = Merge(tmp_path[0], Z_b, Z_c, new_group,aja_Bene,Compet_Matrix,logger)
                                Y = len(new_group)
                                flag1 = np.zeros(Y)
                            else:
                                continue
                            f = 1
                            Z_b,Z_c,new_group = MergeCycle(Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger)
                            break
            if (f == 0):
                flag1[node] = 1
                Y = len(new_group)


        Y = len(new_group)
    return Z_b,Z_c,new_group

def MergeNeighbors(Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger):
    while(1):
        f = 0
        v_1 = -1
        v_2 = -1
        for i in range(len(Z_b)):
            for j in range(len(Z_b)):
                if j != i and f==0:
                    if Z_b[i][j] == 1 and (len(new_group[i]) >= 2 and len(new_group[j]) >= 2) and Z_c[i, j] == 0:
                        f = 1
                        v_1 = i
                        v_2 = j
        if(f):
            Z_b, Z_c, new_group = Merge([v_1, v_2], Z_b, Z_c, new_group,aja_Bene,Compet_Matrix,logger)
            Z_b, Z_c, new_group = MergeCycle(Z_b, Z_c, new_group,aja_Bene,Compet_Matrix,logger)
            Z_b, Z_c, new_group = MergePath(Z_b, Z_c, new_group,aja_Bene,Compet_Matrix,logger)
        else:
            break
    return Z_b,Z_c,new_group






if __name__ == '__main__':
    # log
    if args.target_dir == "":
        args.log_dir = os.path.join(args.outputs_root, get_random_dir_name())
    else:
        args.log_dir = os.path.join(args.outputs_root, args.target_dir)
    setup_seed(seed=args.seed)

    # prepare for learning

    initial_device = torch.device(
        'cuda:{}'.format(args.gpus[0]) if args.gpus != '-1' and torch.cuda.is_available() else 'cpu')
    args.hnet_model_dir = os.path.join(args.log_dir, "hnet_model_saved")
    args.local_hnet_model_dir = os.path.join(args.log_dir, "local_hnet_model_saved")
    args.tensorboard_dir = os.path.join(args.log_dir, "loss")

    logger = construct_log(args)

    if args.dataset == "cifar10":
        args.local_bs = 512
        args.num_users = 10
    elif args.dataset == 'eicu':
        args.num_users = len(args.hospital_id)
        args.n_classes = 2

    logger.info("{}".format(args))
    users_used = [i for i in range(args.num_users)]
    if args.dataset == 'eicu':
        client_id, dataset_train, dataset_valid, dataset_test,dict_users_train,dict_users_test,dict_users_valid = get_dataset(args)
        _, train_dataset, valid_dataset, test_dataset= get_dataset1(
            args)
        client_weights, train_loaders, valid_loaders, test_loaders = get_dataloader(args, train_dataset, valid_dataset,
                                                                                    test_dataset)
        traindata_cls_counts = np.array([[769, 8],
                                         [722, 21],
                                         [739, 6],
                                         [712, 18],
                                         [701, 21],
                                         [361, 16],
                                         [384, 7],
                                         [398, 10],
                                         [405, 10],
                                         [402, 15]]
                                        )
        Data = data(args, dataset_train, dataset_test, dict_users_train, dict_users_test, users_used=users_used,traindata_cls_counts=traindata_cls_counts)
    else:
        dataset_train, dataset_test, dict_users_train, dict_users_test,data_distributions,traindata_cls_counts = get_data(args)
        dict_users_train_tmp = copy.deepcopy(dict_users_train)
        Data = data(args, dataset_train, dataset_test, dict_users_train, dict_users_test, users_used=users_used,traindata_cls_counts=traindata_cls_counts)


    all_results = []
    all_results1= []

    for run in range(5):
        args.runs = run
        results = np.zeros(args.num_users)
        results1 = np.zeros(args.num_users)
        Compet_Matrix = Generate_competetive_graph(args.compete_ratio, args.num_users)
        logger.info('Compet_Matrix:{}'.format(Compet_Matrix))
        if args.train_baseline and args.baseline_type == 'local':
            logger.info(args.baseline_type)
            if args.dataset == 'eicu':
                for usr in range(args.num_users):
                    args.target_usr = usr
                    model = Training_all_eicu(args, logger,Data,client_weights, train_loaders, valid_loaders, test_loaders,
                                              users_used=[usr])
                    model.train()
                    test_loss, test_accuracy, auroc_macro, auprc_macro = model.test(usr)
                    results[usr] = auroc_macro
                    results1[usr] = auprc_macro
                    logger.info(
                        'user {} loss:{} acc:{} auroc:{} auprc:{}'.format(usr, test_loss, test_accuracy, auroc_macro,
                                                                          auprc_macro))
                all_results.append(results)
                all_results1.append(results1)
            else:
                for usr in range(args.num_users):
                    args.target_usr = usr
                    model = Training_all(args, logger, Data, users_used=[usr])
                    model.train()
                    accs, aucs, loss_dict = model.valid(target_usr=usr)
                    logger.info(
                        'user {} acc:{} auc:{} mse:{}'.format(usr, accs[str(usr)], aucs[str(usr)], loss_dict[str(usr)]))

                    if args.dataset == 'eicu':
                        results[usr] = aucs[str(usr)]
                    else:
                        results[usr] = accs[str(usr)]
                all_results.append(results)
            logger.info('results: {},mean_result{}'.format(results, np.mean(results)))
            logger.info('results1: {},mean_result1{}'.format(results1, np.mean(results1)))

        elif args.train_baseline and args.baseline_type == 'fedave':
            logger.info(args.baseline_type)
            if args.dataset == 'cifar10':
                coalition_group = generate_coalition_group(args,Compet_Matrix)
            if args.dataset == 'eicu':
                if  args.num_users == 10:
                    Compet_Matrix = np.array([
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                coalition_group = generate_coalition_group(args,Compet_Matrix)

            logger.info('coalition_group: {}'.format(coalition_group))
            print("----No benefit, no collaboration")
            new_coalition_group = []
            args.train_baseline = False
            for group in coalition_group:
                 if args.dataset == 'eicu':
                     S,_ = SCC(args,group,logger,Data, client_weights, train_loaders, valid_loaders, test_loaders)
                 else:
                     S,_ = SCC(args,group,logger,Data)
                 for s in S:
                     new_coalition_group.append(s)
            args.train_baseline = True
            if args.dataset == 'eicu':
                for group in new_coalition_group:
                    logger.info('group: {}'.format(group))
                    model = Training_all_eicu(args, logger,Data, client_weights, train_loaders, valid_loaders, test_loaders,
                                              users_used=group)
                    model.train()
                    for usr in group:
                        test_loss, test_accuracy, auroc_macro, auprc_macro = model.test(usr)
                        results[usr] = auroc_macro
                        results1[usr] = auprc_macro
                        logger.info('user {} loss:{} acc:{} auroc:{} auprc:{}'.format(usr, test_loss, test_accuracy,
                                                                                      auroc_macro,
                                                                                      auprc_macro))

                all_results.append(results)
                all_results1.append(results1)
                logger.info('results: {},mean_result {}'.format(results, np.mean(results)))
                logger.info('results1: {},mean_result1 {}'.format(results1, np.mean(results1)))

            else:
                for group in new_coalition_group:
                    logger.info('group: {}'.format(group))
                    print('group: {}'.format(group))
                    model = Training_all(args, logger, Data, users_used=group)
                    model.train()

                    accs, aucs, loss_dict = model.valid()
                    for usr in group:
                        args.target_usr = usr
                        if args.dataset == 'eicu':
                            results[usr] = aucs[str(usr)]
                        else:
                            results[usr] = accs[str(usr)]
                        logger.info('user {} acc:{} auc:{} mse:{}'.format(usr, accs[str(usr)], aucs[str(usr)],
                                                                          loss_dict[str(usr)]))
                all_results.append(results)

            result_sum = 0
            for user in range(len(results)):
                result_sum += results[user]
            mean_result = result_sum / len(results)
            logger.info('results: {}, mean_result:{}'.format(results, mean_result))

        elif not args.train_baseline and args.baseline_type == 'fedegoists':
            logger.info(args.baseline_type)
            print(args.baseline_type) #basetype
            args.train_pt = True
            users_used = [i for i in range(args.num_users)]
            results = np.zeros(args.num_users)
            if args.dataset == "eicu":
                if args.num_users == 10:
                    Compet_Matrix = np.array([
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            clique = generate_group(args,Compet_Matrix)#Gc-
            new_group = []
            aja_Bene = np.zeros([args.num_users,args.num_users])

            for group in clique:
                if args.dataset == 'eicu':
                    S,aja_tmp= SCC(args, group, logger, Data, client_weights, train_loaders, valid_loaders, test_loaders)
                else:
                    S,aja_tmp = SCC(args, group, logger, Data)

                for i in range(len(aja_tmp)):
                    for j in range(len(aja_tmp)):
                        if (aja_tmp[i, j] == 1):
                            aja_Bene[i, j] = 1

                for s in S:
                    new_group.append(s)
            # ---------------------------------Benefit_Matrix--------------------------------------
            if args.dataset == 'eicu':
                model = Training_all_eicu(args, logger, Data, client_weights, train_loaders, valid_loaders,
                                          test_loaders,
                                          users_used=users_used)
                if args.way_to_benefit == 'hyper':
                    model.train()
                Benefit_Matrix = model.benefit(users_used, results, results1)
            else:
                model = Training_all(args, logger, Data, users_used=users_used)
                if args.way_to_benefit == 'hyper':
                    model.train()
                Benefit_Matrix = model.benefit(users_used, results)

            logger.info('results: {}, mean_result:{}'.format(results, np.mean(results)))
            logger.info('results1: {}, mean_result1:{}'.format(results1, np.mean(results1)))

            epsilon = args.epsilon
            Benefit_Matrix = np.vstack(Benefit_Matrix.T)
            Benefit_Matrix = Benefit_Matrix * (Benefit_Matrix > epsilon)

            for i in range(len(aja_Bene)):
                for j in range(len(aja_Bene)):
                    if Benefit_Matrix[i,j] >0 and aja_Bene[i,j] == 0:
                        aja_Bene[i,j] = 1

            Y = len(new_group)
            #construct Zb and Zc
            Z_b = np.zeros([Y, Y])
            for i in range(Y):
                for j in range(Y):
                    if j==i:
                        continue
                    f = 0
                    for m in new_group[i]:
                        for n in new_group[j]:
                            if aja_Bene[m][n] ==1:
                                f = 1
                    if f:
                        Z_b[i][j] = 1

            Z_c = np.zeros([Y, Y])
            for i in range(Y):
                for j in range(i+1,Y):
                    f = 0
                    for m in new_group[i]:
                        for n in new_group[j]:
                            if Compet_Matrix[m][n]==1:
                                f = 1
                    if f:
                        Z_c[i][j] = 1
                        Z_c[j][i] = 1
            logger.info("Z_b{}".format(Z_b))
            logger.info("Z_c{}".format(Z_c))
            print(Z_b)
            print(Z_c)
            logger.info("new_group{}".format(new_group))
            print(new_group)


            Z_b,Z_c,new_group = MergeCycle(Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger)
            Z_b,Z_c,new_group = MergePath(Z_b, Z_c, new_group,aja_Bene,Compet_Matrix,logger)
            Z_b,Z_c,new_group = MergeNeighbors(Z_b,Z_c,new_group,aja_Bene,Compet_Matrix,logger)

            pai = new_group


            args.train_pt = False

            for user in range(args.num_users):
                args.target_usr = user
                contribute_users = []
                for group in pai:
                    if user in group:
                        users_used = group
                args.baseline_type = 'fedegoists'
                if args.dataset == 'eicu':
                    model = Training_all_eicu(args, logger, Data, client_weights, train_loaders, valid_loaders,
                                              test_loaders,
                                              users_used=users_used)
                    model.train_new(user, users_used)
                    test_loss, test_accuracy, auroc_macro, auprc_macro = model.test(user)
                    logger.info('user:{},auroc_macro:{}'.format(user, auroc_macro))
                    results[user] = auroc_macro
                    results1[user] = auprc_macro

                else:
                    model = Training_all(args, logger, Data, users_used)
                    model.train_new(user, users_used)
                    acc, auc, loss = model.personal_test(user)

                    if args.dataset == 'eicu':
                        results[user] = auc
                    else:
                        results[user] = acc
                logger.info('user:{},results: {},results1:{}'.format(user, results, results1))

            result_sum = 0

            all_results.append(results)
            all_results1.append(results1)
            for user in range(len(results)):
                result_sum += results[user]
            mean_result = result_sum / len(results)

            logger.info('results:{}, mean_result: {}'.format(results, mean_result))
            logger.info('results1: {},mean_result1 {}'.format(results1, np.mean(results1)))

            args.baseline_type = 'fedegoists'
            args.train_baseline = False
            args.train_pt = True

    logger.info('all_results: {}'.format(all_results))
    mean_col = np.mean(np.array(all_results), axis=0)
    std_dev_col = np.std(np.array(all_results), axis=0)
    logger.info('mean_value {},std_value{}'.format(mean_col, std_dev_col))
    logger.info('all_mean_value{},all_std_value{}'.format(np.mean(np.mean(np.array(all_results), axis=1)),
                                                          np.std(np.mean(np.array(all_results), axis=1))))

    if args.dataset == 'eicu':
        logger.info('all_results1: {}'.format(all_results1))
        mean_col1 = np.mean(np.array(all_results1), axis=0)
        std_dev_col1 = np.std(np.array(all_results1), axis=0)
        logger.info('mean_value1 {},std_value1{}'.format(mean_col1, std_dev_col1))
        logger.info('all_mean_value1 {},all_std_value1 {}'.format(np.mean(np.mean(np.array(all_results1), axis=1)),
                                                              np.std(np.mean(np.array(all_results1), axis=1))))


