from __future__ import print_function
import argparse
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from datasets import MarketDatasetTest, MarketDatasetTrain
from evaluation import Evaluation
from glob import glob
from layers import DropoutShared, L2Normalization
from loss import HistogramLoss
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from visualizer import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='batch size for train, default=128')
parser.add_argument('--batch_size_test', type=int, default=64, help='batch size for test and query dataloaders for market dataset, default=64')
parser.add_argument('--checkpoints_path', default='.', help='folder to output model checkpoints, default="."')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='probability of dropout, default=0.5')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=1e-4')
parser.add_argument('--lr_init', type=float, default=1e-2, help='learning rate for first stage of training only fc layer, default=1e-2')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--market', action='store_true', help='calculate rank1 and mAP on Market dataset; dataroot should contain folders "bounding_box_train", "bounding_box_test", "query"')
parser.add_argument('--nbins', default=150, type=int, help='number of bins in histograms, default=150')
parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train, default=150')
parser.add_argument('--nworkers', default=10, type=int, help='number of data loading workers, default=10')
parser.add_argument('--visdom_port', type=int, help='port for visdom visualization')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.checkpoints_path)
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manual_seed)
    
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

vis = Visualizer(opt.checkpoints_path, opt.visdom_port)
    
paths = {
    'train': 'bounding_box_train/*.jpg',
    'test': 'bounding_box_test/*.jpg',
    'query': 'query/*.jpg',
}

def create_df(dir, size=-1):
    df_paths = glob(dir)
    df = pd.DataFrame({'path': df_paths})
    return df[:size]

def create_market_df(x):
        df = create_df(os.path.join(opt.dataroot, paths[x]))
        df['label'] = df.path.apply(lambda x: int(x.split('/')[-1].split('_')[0]))
        df['camera'] = df.path.apply(lambda x: int(x.split('/')[-1].split('_')[1].split('s')[0].split('c')[1]))
        df['name'] = df.path.apply(lambda x: x.split('/')[-1])

        return df

if opt.market:
    market_df = create_market_df('train')
else:
    market_df = create_df(os.path.join(opt.dataroot))
    
labels_encoder = LabelEncoder().fit(market_df['label'])
data_transform = transforms.Compose([
    transforms.Scale([256, 256]),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = MarketDatasetTrain(market_df['path'], 
                             market_df['label'], 
                             labels_encoder, 
                             data_transform, 
                             opt.batch_size)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

if opt.market:
    market_dfs_test = {
        x: create_market_df(x) for x in ['test', 'query']
    }
    
    data_transform_test = transforms.Compose([
        transforms.Scale([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    datasets_test = {
        x: MarketDatasetTest(
            market_dfs_test[x]['path'], 
            transform=data_transform_test,
        ) for x in ['test', 'query']
    }

    dataloaders_test = {
        x: DataLoader(
            datasets_test[x], 
            batch_size=opt.batch_size_test,              
            shuffle=False, 
            num_workers=opt.nworkers
        ) for x in datasets_test.keys()
    }

    evaluation = Evaluation(market_dfs_test['test'], market_dfs_test['query'], dataloaders_test['test'], dataloaders_test['query'], opt.cuda)
    
criterion = HistogramLoss(opt.nbins, opt.cuda)

def train(optimizer, scheduler, epoch_start, epoch_end):
    for epoch in range(epoch_start, epoch_end + 1):
        scheduler.step()
        model.train(True)
        running_loss = .0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.squeeze()), Variable(labels.squeeze())

            if opt.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

        epoch_loss = running_loss / len(dataloader)

        vis.quality('Loss', {'Loss': epoch_loss}, epoch, opt.nepoch + 1)

        if opt.market:
            if epoch % 5 == 0:
                model.train(False)
                ranks, mAP = ranks, mAP = evaluation.ranks_map(model, 50)
                vis.quality('Rank1 and mAP', {'Rank1': ranks[1], 'mAP': mAP}, epoch, opt.nepoch + 1)

        if epoch % 10 == 0:
            torch.save(model, '{}/finetuned_histogram_e{}.pt'.format(opt.checkpoints_path, epoch))

model = models.resnet34(pretrained=True)  
for param in model.parameters():
    param.requires_grad = False
    
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential()
model.fc.add_module('shared_dropout', DropoutShared(p=.5, use_gpu=True))
model.fc.add_module('fc', nn.Linear(num_ftrs, 512))
model.fc.add_module('l2normalization', L2Normalization())

if opt.cuda:
    model = model.cuda()
    
print(model)
print('Train only last layer\n')

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr_init)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
train(optimizer, scheduler, 1, 20)

print('Train all layers\n')
for param in model.parameters():
    param.requires_grad = True
    
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
train(optimizer, scheduler, 21, opt.nepoch)
