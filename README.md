# FedEgoists

## The implementation of Competitors

---
### The description of all files
1. main.py: the main function for all experiments;
2. benefit.py: this script contains the function to get the benefit graph and train function for experiments;
3. data.py: this script executes assigning data to participants;
4. graph.py: this script contains a variety of related graph processing algorithms;
5. models.py: this script defines model structures;
6. solvers.py: this script contains optimization for learning the whole Pareto Front;
7. utils_data.py: this script pre-processes all data set which will be used for the following training and evaluating;
8. utils_func.py: the needed extra functions;
9. models.py: this script defines all necessary model structures;
10. utils_sampling.py: this script is used for generating non i.i.d data for experiments;
11. tarjan.py: this script is used for finding the SCC;
12. cached_dataset.py: this script is used for getting the eICU
---

## Preparations

### Construct  Environment
python 3.8.13, the needed environment libraries are in requirements.txt.
### Datasets
eICU dataset needs approval when researchers need to have access to it. We get the dataset from [eICU](https://eicu-crd.mit.edu/). Then we process it follow by [wns823](https://github.com/wns823/medical_federated/blob/main/ehr_federated/README.md).


## Get Started

### Parameters Description in main.py for Running experiments
1. dataset: the needed dataset for running experiments;
2. total_hnet_epoch: the num of epoch for training the Pareto Front;
3. total_ray_epoch: the num of epoch for training the direction vector;
4. personal_epoch: the num of epoch for personal training;
5. lr: learning rate for running experiments;
6. lr_prefer: learning rate for updating the direction vector;
7. gpus: the GPU device;
8. compete_ratio: the compete ratio for generating competing graph;
9. epsilon: the benefit graph threshold.

### Example CIFAR10 PAT Experiment

```
python main.py --gpus 0 --dataset cifar10 --partition kdd --shard_per_user 2 --total_hnet_epoch 100 --total_ray_epoch 100 --seed 22 --lr 0.01 --lr_prefer 0.01  --baseline_type fedcolcompetitors --way_to_benefit hyper --personal_init_epoch 100 --personal_epoch 100 --total_epoch 100 --compete_ratio 0.2 --epsilon 0.095 --optim sgd --lamda 15 --server_learning_rate 0.01 --beta 1
```

### Example CIFAR10 DIR Experiment

```
python main.py --gpus 0 --dataset cifar10 --partition labeldir --total_hnet_epoch 100 --total_ray_epoch 100 --seed 22 --lr 0.01 --lr_prefer 0.01  --baseline_type fedcolcompetitors --way_to_benefit hyper --personal_init_epoch 1000 --personal_epoch 1000 --total_epoch 1000 --compete_ratio 0.2 --epsilon 0.095 --optim sgd --lamda 15 --server_learning_rate 0.001 --beta 1
```
### Example CIFAR100 PAT Experiment

```
python main.py --gpus 0 --dataset cifar100 --partition kdd --shard_per_user 20 --total_hnet_epoch 400 --total_ray_epoch 400 --seed 33 --lr 0.005 --lr_prefer 0.1  --baseline_type fedcolcompetitors --way_to_benefit hyper --personal_init_epoch 1000 --personal_epoch 1000 --total_epoch 1000 --compete_ratio 0.2 --epsilon 0.15 --optim sgd --lamda 15 --server_learning_rate 0.0005 --beta 0.8 --n_classes 100
```

### Example CIFAR100 DIR Experiment

```
python main.py --gpus 0 --dataset cifar100 --partition labeldir --total_hnet_epoch 400 --total_ray_epoch 400 --seed 33 --lr 0.005 --lr_prefer 0.01  --baseline_type fedcolcompetitors --way_to_benefit hyper --personal_init_epoch 1000 --personal_epoch 1000 --total_epoch 1000 --compete_ratio 0.2 --epsilon 0.095 --optim sgd --lamda 15 --server_learning_rate 0.005 --beta 0.8 --n_classes 100
```

### Example eICU Experiment

```
python main.py --gpus 0 --dataset eicu --total_hnet_epoch 100 --total_ray_epoch 100 --seed 1 --lr 0.00001 --lr_prefer 0.0001 --batch_size 256 --baseline_type fedcolcompetitors --way_to_benefit hyper --total_epoch 100 --task "mort_48h" --personal_init_epoch 100 --personal_epoch 100 --epsilon 0.085
```

## Reference

Cui, S.; Liang, J.; Pan, W.; Chen, K.; Zhang, C.; and Wang, F. 2022. Collaboration equilibrium in federated learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (*KDD’22*)*,241–251.
