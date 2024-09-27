import json, pickle, enum, copy, os, random, sys, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Dict

class CachedDataset(Dataset): # Dataset
    def __init__(
        self,
        hospital_id,
        unitstays,
        data_path,
        task
    ):
        self.hospital_id = hospital_id
        self.unitstay = unitstays
        self.load_path = f"{data_path}/eicu-2.0/federated_preprocessed_data/cached_data"
        self.task = task

    def __len__(self): 
        return len(self.unitstay)

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', None
                'ts', [batch_size, sequence_length, 165]
                'statics', [batch_size, 15]
                'next_timepoint',
                'next_timepoint_was_measured',
                'disch_24h', [batch_size, 10]
                'disch_48h', [batch_size, 10]
                'Final Acuity Outcome', [batch_size, 12]
                'ts_mask',
                'tasks_binary_multilabel', [batch_size, 3]
        """

        unitstay = self.unitstay[item]

        unitstay_path = os.path.join( self.load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)
        X = tensors['ts'],tensors['statics']
        if self.task in ['disch_24h', 'disch_48h']:
            labels = tensors[self.task]
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            isnan = (labels == -9223372036854775808)
            Y = torch.where(isnan, torch.zeros_like(labels), labels)

        elif self.task in ['mort_24h', 'mort_48h', 'LOS']:
            t = ['mort_24h', 'mort_48h', 'LOS'].index(self.task)
            labels = tensors['tasks_binary_multilabel'][t]  # .to(device)

            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            isnan = torch.isnan(labels)
            Y = torch.where(isnan, torch.zeros_like(labels), labels)

        else:
            labels = tensors[self.task]
            if labels.dim() == 0:
                Y = labels.unsqueeze(0)
        # return tensors
        return X,Y

class CachedDataset1(Dataset): # Dataset
    def __init__(
        self,
        hospital_id,
        unitstays,
        data_path
    ):
        self.hospital_id = hospital_id
        self.unitstay = unitstays
        self.load_path = f"{data_path}/eicu-2.0/federated_preprocessed_data/cached_data"

    def __len__(self):
        return len(self.unitstay)

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', None
                'ts', [batch_size, sequence_length, 165]
                'statics', [batch_size, 15]
                'next_timepoint',
                'next_timepoint_was_measured',
                'disch_24h', [batch_size, 10]
                'disch_48h', [batch_size, 10]
                'Final Acuity Outcome', [batch_size, 12]
                'ts_mask',
                'tasks_binary_multilabel', [batch_size, 3]
        """

        unitstay = self.unitstay[item]

        unitstay_path = os.path.join( self.load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)

        return tensors