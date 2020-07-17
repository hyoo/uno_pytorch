"""
   minimal implementation of uno in pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import List
from time import time


class FeatureModel(nn.Module):
    def __init__(self, input_shape: int, name: str = '', dense_layers: List[int] = [1024, 1024],
                 dropout_rate: float = 0):
        super(FeatureModel, self).__init__()

        self.model = nn.Sequential()
        prev_dim = input_shape
        for i, layer in enumerate(dense_layers):
            hidden = nn.Linear(prev_dim, layer)
            self.model.add_module(f'{name}_dense_{i}', hidden)
            prev_dim = layer
            self.model.add_module(f'{name}_relu_{i}', nn.ReLU())
            if dropout_rate > 0:
                self.model.add_module(f'{name}_dr_{i}', nn.Dropout(dropout_rate))
    
    def forward(self, data):
        return self.model(data)


class UnoModel(nn.Module):
    def __init__(self, gene_latent_dim: int = 256, drug_latent_dim: int = 1024, dense_layers: List[int] = [1024, 512, 256, 128, 64],
                 dropout_rate: float = 0):
        super(UnoModel, self).__init__()

        self.gene_net = FeatureModel(942, 'gene', [628, 512, gene_latent_dim], dropout_rate=dropout_rate)
        self.drug_net = FeatureModel(5270, 'drug', [3514, 2048, drug_latent_dim], dropout_rate=dropout_rate)
        self.resp_net = nn.Sequential()

        prev_dim = gene_latent_dim + drug_latent_dim
        for i, width in enumerate(dense_layers):
            self.resp_net.add_module(f'resp_{i}', nn.Linear(prev_dim, width))
            prev_dim = width
            self.resp_net.add_module(f'resp_relu_{i}', nn.ReLU())
            if dropout_rate > 0:
                self.resp_net.add_module(f'resp_dropout_{i}', nn.Dropout(dropout_rate))

        self.resp_net.add_module('resp_dense_out', nn.Linear(prev_dim, 2))

    def forward(self, gene, drug):
        return self.resp_net(torch.cat((self.gene_net(gene), self.drug_net(drug)), dim=1))


def load_data():
    # download from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
    datafile = '/home/hsyoo/CANDLE/Benchmarks/Pilot1/Uno/top_21_auc_1fold.uno.h5'
    kwargs = {'num_workers': 3, 'pin_memory': True}

    y_train = torch.tensor(pd.read_hdf(datafile, 'y_train')['AUC'].apply(lambda x: 1 if x < 0.5 else 0).values)
    x_train_0 = torch.tensor(pd.read_hdf(datafile, 'x_train_0').values)
    x_train_1 = torch.tensor(pd.read_hdf(datafile, 'x_train_1').values)

    _, counts = np.unique(y_train, return_counts=True)
    weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = torch.utils.data.TensorDataset(y_train, x_train_0, x_train_1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=512, shuffle=False, **kwargs
    )

    y_val = torch.tensor(pd.read_hdf(datafile, 'y_val')['AUC'].apply(lambda x: 1 if x < 0.5 else 0).values)
    x_val_0 = torch.tensor(pd.read_hdf(datafile, 'x_val_0').values)
    x_val_1 = torch.tensor(pd.read_hdf(datafile, 'x_val_1').values)
    
    val_dataset = torch.utils.data.TensorDataset(y_val, x_val_0, x_val_1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1024, shuffle=False, **kwargs
    )

    return train_loader, val_loader


def main():
    init = time()
    device = torch.device("sycl")
    train_loader, val_loader = load_data()

    model = UnoModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6)
    ed = time()
    print(f"Loading script and data:  {ed-init}")

    for epoch in range(1, 11):
        # train
        st = time()
        model.train()
        for batch_idx, (target, gene, drug) in enumerate(train_loader):
            target, gene, drug = target.to(device), gene.to(device), drug.to(device)
            optimizer.zero_grad()
            output = model(gene, drug)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        ed = time()
        print(f"Elapsed time for training: {ed-st}")
        # eval
        st = time()
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for target, gene, drug in val_loader:
                target, gene, drug = target.to(device), gene.to(device), drug.to(device)
                output = model(gene, drug)
                val_loss = F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                target, pred = target.cpu(), pred.cpu()
                accuracy = accuracy_score(target, pred)
                precision = precision_score(target, pred, average=None)
                recall = recall_score(target, pred, average=None)
        lr = optimizer.param_groups[0]['lr']
        ed = time()
        print(f"Elasped time for eval: {ed-st}")
        scheduler.step(val_loss)
        print(f'Epoch: {epoch:03} {(ed-init):03} sec lr: {lr:.6f} loss: {loss:.5f} val_loss: {val_loss:.4f} accuracy: {accuracy:.4f} precision neg: {precision[0]:.4f} pos: {precision[1]:.4f} recall neg: {recall[0]:.4f} pos: {recall[1]:.4f}')


if __name__ == '__main__':
    main()
