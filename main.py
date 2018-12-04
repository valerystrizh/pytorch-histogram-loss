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

from functools import partial
from glob import glob
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from config_reader import config_reader
from datasets import ImageDataset
from evaluation import Evaluation
from layers import L2Normalization
from losses import HistogramLoss
from samplers import MarketSampler
from visualizer import Visualizer

opt = config_reader()
print(opt)

try:
    os.makedirs(opt['checkpoints_path'])
except OSError:
    pass

if opt['manual_seed'] is None:
    opt['manual_seed'] = random.randint(1, 10000)
print("Random Seed: ", opt['manual_seed'])
random.seed(opt['manual_seed'])
torch.manual_seed(opt['manual_seed'])
if opt['cuda']:
    torch.cuda.manual_seed_all(opt['manual_seed'])
    
if torch.cuda.is_available() and not opt['cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

vis = Visualizer(opt['checkpoints_path'], opt['visdom_port'])

def create_df(dataroot, size=-1):
    df_paths = glob(dataroot)
    df = pd.DataFrame({'path': df_paths})
    df['label'] = df.path.apply(lambda x: int(x.split('/')[-1].split('_')[0]))
    return df[:size]

if not opt['market']:
    df_train = create_df(os.path.join(opt['dataroot'], '*.jpg'))
else:
    def create_market_df(x):
        df = create_df(os.path.join(opt['dataroot'], paths[x]))
        df['camera'] = df.path.apply(lambda x: int(x.split('/')[-1].split('_')[1].split('s')[0].split('c')[1]))
        df['name'] = df.path.apply(lambda x: x.split('/')[-1])
        return df
    
    paths = {
        'train': 'bounding_box_train/*.jpg',
        'test': 'bounding_box_test/*.jpg',
        'query': 'query/*.jpg',
    }
    
    df_train = create_market_df('train')
    dfs_test = {
        x: create_market_df(x) for x in ['test', 'query']
    }
    
    data_transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    datasets_test = {
        x: ImageDataset(
            dfs_test[x]['path'], 
            transform=data_transform_test,
            is_train=False
        ) for x in ['test', 'query']
    }

    dataloaders_test = {
        x: DataLoader(
            datasets_test[x], 
            batch_size=opt['batch_size_test'],              
            shuffle=False, 
            num_workers=opt['nworkers']
        ) for x in datasets_test.keys()
    }

    evaluation = Evaluation(dfs_test['test'], dfs_test['query'], dataloaders_test['test'], dataloaders_test['query'], opt['cuda'])
    
data_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageDataset(df_train['path'], data_transform, True, df_train['label'])
sampler = MarketSampler(df_train['label'], opt['batch_size'])
dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=opt['nworkers'])

def train(optimizer, criterion, scheduler, epoch_start, epoch_end):
    for epoch in range(epoch_start, epoch_end + 1):
        scheduler.step()
        model.train(True)
        running_loss = .0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.squeeze()), Variable(labels.squeeze())

            if opt['cuda']:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

        epoch_loss = running_loss / len(dataloader)

        vis.quality('Loss', {'Loss': epoch_loss}, epoch, opt['nepoch'])

        if opt['market']:
            if (epoch - 1) % 5 == 0:
                model.train(False)
                ranks, mAP = ranks, mAP = evaluation.ranks_map(model, 2)
                vis.quality('Rank1 and mAP', {'Rank1': ranks[1], 'mAP': mAP}, epoch, opt['nepoch'])

        if (epoch - 1) % 10 == 0:
            torch.save(model, '{}/finetuned_histogram_e{}.pt'.format(opt['checkpoints_path'], epoch))

model = models.resnet34(pretrained=True)  
for param in model.parameters():
    param.requires_grad = False
    
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential()
if opt['dropout_prob'] > 0:
    model.fc.add_module('dropout', nn.Dropout(opt['dropout_prob']))
model.fc.add_module('fc', nn.Linear(num_ftrs, 512))
model.fc.add_module('l2normalization', L2Normalization())
if opt['cuda']:
    model = model.cuda()
print(model)

criterion = HistogramLoss(num_steps=opt['nbins'], cuda=opt['cuda'])

if opt['nepoch_fc'] > 0:
    print('\nTrain fc layer\n')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt['lr_fc'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train(optimizer, criterion, scheduler, 1, opt['nepoch_fc'])

print('\nTrain all layers\n')
for param in model.parameters():
    param.requires_grad = True
    
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt['lr'])
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
train(optimizer, criterion, scheduler, opt['nepoch_fc'] + 1, opt['nepoch'])
