"""
   minimal implementation of uno in pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import apex
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import List
from time import time
import argparse


class FeatureModel(nn.Module):
    def __init__(self, input_shape: int, name: str = '', dense_layers: List[int] = [256, 128],
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
    def __init__(self,
                 gene_layers: List[int] = [942, 512, 256],
                 drug_layers: List[int] = [5270, 2048, 1024],
                 dense_layers: List[int] = [1024, 512, 256, 128, 64],
                 dropout_rate: float = 0.,
                 final_dim: int = 1):
        super(UnoModel, self).__init__()

        self.gene_net = FeatureModel(942, 'gene', gene_layers, dropout_rate=dropout_rate)
        self.drug_net = FeatureModel(5270, 'drug', drug_layers, dropout_rate=dropout_rate)
        self.resp_net = nn.Sequential()

        prev_dim = gene_layers[-1] + drug_layers[-1]
        for i, width in enumerate(dense_layers):
            self.resp_net.add_module(f'resp_{i}', nn.Linear(prev_dim, width))
            prev_dim = width
            self.resp_net.add_module(f'resp_relu_{i}', nn.ReLU())
            if dropout_rate > 0:
                self.resp_net.add_module(f'resp_dropout_{i}', nn.Dropout(dropout_rate))

        self.resp_net.add_module('resp_dense_out', nn.Linear(prev_dim, final_dim))

    def forward(self, gene, drug):
        return self.resp_net(torch.cat((self.gene_net(gene), self.drug_net(drug)), dim=1))


def load_data(args):
    # download from http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
    # datafile = '/home/hsyoo/CANDLE/Benchmarks/Pilot1/Uno/top_21_auc_1fold.uno.h5'
    kwargs = {'num_workers': 3, 'pin_memory': True}
    dtype = eval(f'torch.{args.dtype}')

    if args.mode == 'cls':
        y_train = torch.tensor(pd.read_hdf(args.data, 'y_train')['AUC'].apply(lambda x: 1 if x < 0.5 else 0).values)
    else:
        y_train = torch.tensor(pd.read_hdf(args.data, 'y_train')['AUC'].values, dtype=dtype)
    x_train_0 = torch.tensor(pd.read_hdf(args.data, 'x_train_0').values, dtype=dtype)
    x_train_1 = torch.tensor(pd.read_hdf(args.data, 'x_train_1').values, dtype=dtype)

    if args.mode == 'cls':
        y_val = torch.tensor(pd.read_hdf(args.data, 'y_val')['AUC'].apply(lambda x: 1 if x < 0.5 else 0).values)
    else:
        y_val = torch.tensor(pd.read_hdf(args.data, 'y_val')['AUC'].values, dtype=dtype)
    x_val_0 = torch.tensor(pd.read_hdf(args.data, 'x_val_0').values, dtype=dtype)
    x_val_1 = torch.tensor(pd.read_hdf(args.data, 'x_val_1').values, dtype=dtype)

    if args.mode == 'cls':
        _, counts = np.unique(y_train, return_counts=True)
        weights = 1. / torch.tensor(counts, dtype=torch.float)
        sample_weights = weights[y_train]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_dataset = torch.utils.data.TensorDataset(y_train, x_train_0, x_train_1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.batch_size, shuffle=False, **kwargs
        )

        val_dataset = torch.utils.data.TensorDataset(y_val, x_val_0, x_val_1)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    else:
        train_dataset = torch.utils.data.TensorDataset(y_train, x_train_0, x_train_1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs
        )

        val_dataset = torch.utils.data.TensorDataset(y_val, x_val_0, x_val_1)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False, **kwargs
        )

    return train_loader, val_loader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'sycl'], help='Target device')
    parser.add_argument('--data', default='./top_21_auc_1fold.uno.h5', help='Datafile location')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='Epochs')
    parser.add_argument('--batch_size', '-z', default=32, type=int, help='Batch Size')
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float, help='Learning Rate')
    parser.add_argument('--mode', default='reg', choices=['reg', 'cls'], help='Regression or Classification')
    parser.add_argument('--apex', default=None, choices=[None, 'O0', 'O1', 'O2'])
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def main():
    args, _ = parse_arguments()
    device = torch.device(args.device)

    init = time()
    train_loader, val_loader = load_data(args)

    model = UnoModel(gene_layers=[1000, 1000, 1000],
                     drug_layers=[1000, 1000, 1000],
                     dense_layers=[1000, 1000, 1000, 1000, 1000],
                     dropout_rate=0.1,
                     final_dim=2 if args.mode == 'cls' else 1
                     ).to(device)
    summary(model, input_size=[(args.batch_size, 942), (args.batch_size, 5270)])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-6)

    if args.apex is not None:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.apex)

    ed = time()
    print(f"Loading script and data:  {(ed-init):.2f}")

    for epoch in range(1, (args.epochs + 1)):
        # train
        st = time()
        model.train()
        for _, (target, gene, drug) in enumerate(train_loader):
            target, gene, drug = target.to(device), gene.to(device), drug.to(device)
            optimizer.zero_grad()
            output = model(gene, drug)
            if args.mode == 'cls':
                loss = F.cross_entropy(output, target)
            else:
                loss = F.mse_loss(output.reshape(-1), target)

            if args.apex is None:
                loss.backward()
            else:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for target, gene, drug in val_loader:
                target, gene, drug = target.to(device), gene.to(device), drug.to(device)
                output = model(gene, drug)
                if args.mode == 'cls':
                    val_loss = F.cross_entropy(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    target, pred = target.cpu(), pred.cpu()
                    accuracy = accuracy_score(target, pred)
                    precision = precision_score(target, pred, average=None)
                    recall = recall_score(target, pred, average=None)
                else:
                    val_loss = F.mse_loss(output.reshape(-1), target)
        lr = optimizer.param_groups[0]['lr']
        ed = time()
        scheduler.step(val_loss)
        if args.mode == 'cls':
            print(f'Epoch: {epoch:03}, elapsed: {(ed-st):.2f} sec(s), lr: {lr:.6f}, loss: {loss:.5f}, val_loss: {val_loss:.4f}, accuracy: {accuracy:.4f}, precision neg: {precision[0]:.4f}, pos: {precision[1]:.4f}, recall neg: {recall[0]:.4f}, pos: {recall[1]:.4f}')
        else:
            print(f'Epoch: {epoch:03}, elasped: {(ed-st):.2f} sec(s), lr: {lr:.6f}, loss: {loss:.5f}, val_loss: {val_loss:.4f}')


if __name__ == '__main__':
    main()
