import numpy as np
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class ImageDatasetTest(Dataset):
    def __init__(self, paths, transform):
        self.img_paths = np.array(paths)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.transform(Image.open(img_path))
        return img
        
class ImageDatasetTrain(Dataset):
    def __init__(self, paths, labels, transform):
        self.img_paths = np.array(paths)
        self.labels = LabelEncoder().fit_transform(labels)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.float))
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.transform(Image.open(img_path))
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.img_paths)
    
