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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSizeTest', type=int, default=64, help='batch size for test and query dataloaders')
parser.add_argument('--batchSizeTrain', type=int, default=128, help='batch size for train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nbins', default=150, type=int, help='number of bins in histograms')
parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train')
parser.add_argument('--nworkers', default=10, type=int, help='number of data loading workers')
parser.add_argument('--out', default='.', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

paths = {
    'train': 'bounding_box_train/*.jpg',
    'test': 'bounding_box_test/*.jpg',
    'query': 'query/*.jpg',
}

def create_market_df(x):
    df_paths = glob(os.path.join(opt.dataroot, paths[x]))
    df = pd.DataFrame({'path': df_paths})
    df['label'] = df.path.apply(lambda x: int(x.split('/')[-1].split('_')[0]))
    df['camera'] = df.path.apply(lambda x: int(x.split('/')[-1].split('_')[1].split('s')[0].split('c')[1]))
    df['name'] = df.path.apply(lambda x: x.split('/')[-1])
    
    return df

market_dfs = {
    x: create_market_df(x) for x in paths.keys()
}

train_labels_encoder = LabelEncoder().fit(market_dfs['train']['label'])

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

datasets = {
    x: MarketDatasetTest(
        market_dfs[x]['path'], 
        transform=data_transforms['test'],
    ) for x in ['test', 'query']
}

dataloaders = {
    x: DataLoader(
        datasets[x], 
        batch_size=opt.batchSizeTest,              
        shuffle=False, 
        num_workers=opt.nworkers
    ) for x in datasets.keys()
}

datasets['train'] = MarketDatasetTrain(market_dfs['train']['path'], 
                                       market_dfs['train']['label'], 
                                       train_labels_encoder, 
                                       data_transforms['train'], 
                                       opt.batchSizeTrain)
dataloaders['train'] = DataLoader(datasets['train'], shuffle=True, batch_size=1)

evaluation = Evaluation(market_dfs['test'], market_dfs['query'])

model = models.resnet34(pretrained=True)  
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential()
model.fc.add_module('shared_dropout', DropoutShared(p=.2, use_gpu=True))
model.fc.add_module('fc', nn.Linear(num_ftrs, 512))
model.fc.add_module('l2normalization', L2Normalization())

if opt.cuda:
    model = model.cuda()
    
print(model)
    
criterion = HistogramLoss(opt.nbins, opt.cuda)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

since = time.time()
losses = []
rank1maps = []

for epoch in range(1, opt.nepoch + 1):
    print('Epoch {}/{}'.format(epoch, opt.nepoch))
    scheduler.step()
    model.train(True)
    running_loss = .0

    for data in dataloaders['train']:
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

    epoch_loss = running_loss / len(dataloaders['train'])
    losses.append((epoch, epoch_loss))
    print('Loss: {:.6f}'.format(epoch_loss))

    if epoch % 5 == 0:
        model.train(False)
        ranks, mAP = evaluation.ranks_map(model, dataloaders, opt.cuda)
        rank1maps.append((epoch, ranks[1], mAP))
        print('rank1: {:.4f} mAP: {:.4f}'.format(ranks[1], mAP))
        
    if epoch % 10 == 0:
        torch.save(model, '{}/finetuned_histogram_e{}.pt'.format(opt.out, opt.nepoch))
        
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

np.save('{}/loss_statistics'.format(opt.out), np.array(losses))
np.save('{}/rank1map_statistics'.format(opt.out), np.array(rank1maps))
