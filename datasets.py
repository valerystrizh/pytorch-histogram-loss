import numpy as np
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, paths, transform, is_train, labels=None):
        self.img_paths = np.array(paths)
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            self.labels = LabelEncoder().fit_transform(labels)
            self.labels = torch.from_numpy(np.array(self.labels, dtype=np.float))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.transform(Image.open(img_path))
        if self.is_train:
            label = self.labels[index]
            return img, label
        else:
            return img
        
    def __len__(self):
        return len(self.img_paths)
    
