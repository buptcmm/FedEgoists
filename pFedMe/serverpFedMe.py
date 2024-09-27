import torch
import os

from torch.utils.data import Dataset

from pFedMe.userpFedMe import UserpFedMe
from pFedMe.serverbase import Server
from pFedMe.model_utils import read_data, read_user_data
import numpy as np
 
# Implementation for pFedMe Server
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class pFedMe(Server):
    def __init__(self,logger,dataset_train,dict_users_train,dataset_test,dict_users_test,device,
                 dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate,):
        super().__init__(logger,device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users)

        # Initialize data for all  users
        # data = read_data(dataset)
        # total_users = len(data[0])
        self.K = K
        self.logger = logger
        self.personal_learning_rate = personal_learning_rate
        train_dataset = {}
        test_dataset = {}

        for user_id in range(num_users):
            # Create training data loader for this user
            train_dataset[user_id] = DatasetSplit(dataset_train, dict_users_train[user_id])
            # Create test data loader for this user
            test_dataset[user_id] = DatasetSplit(dataset_test, dict_users_test[user_id])

        for i in range(num_users):
            # id, train , test = read_user_data(i, data, dataset)
            user = UserpFedMe(device, i, train_dataset[i], test_dataset[i], model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.logger.info("-------------Round number: {}-------------".format(glob_iter))
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.logger.info("Evaluate global model")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()

            if self.dataset=='eicu':
                self.auc_persionalized_model()



        #print(loss)
        # self.save_results()
        # self.save_model()
    
  
