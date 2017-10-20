import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

class MarketDatasetTest(Dataset):
    def __init__(self, paths, transform):
        self.img_paths = np.array(paths)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.transform(Image.open(img_path))

        return img
        
class MarketDatasetTrain(Dataset):
    def __init__(self, paths, labels, le, transform, batch_size=64):
        self.img_paths = np.array(paths)
        self.labels = np.array(le.transform(labels), dtype=np.float)
        self.labels_unique = np.unique(self.labels)
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.img_paths) // self.batch_size
    
    def __getitem__(self,index):
        labels_in_batch = set()
        inds = np.array([], dtype=np.int)
        
        while inds.shape[0] < self.batch_size:
            sample_label = np.random.choice(self.labels_unique)
            if sample_label in labels_in_batch:
                continue
                
            labels_in_batch.add(sample_label)
            sample_in_labels = np.in1d(self.labels, sample_label)
            subsample_size = np.random.choice(range(5, 11))
            subsample = np.random.permutation(np.argwhere(sample_in_labels).reshape(-1))
            subsample = subsample[:subsample_size]
            inds = np.append(inds, subsample)
            
        inds = inds[:self.batch_size]
        subsamples_lables = torch.from_numpy(self.labels[inds])
        subsamples_imgs = self.img_paths[inds]
        subsamples_imgs = torch.stack([self.transform(Image.open(img)) for img in subsamples_imgs])
        
        return subsamples_imgs, subsamples_lables 