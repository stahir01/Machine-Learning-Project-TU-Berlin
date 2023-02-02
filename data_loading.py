import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose

class ISBIEMSegDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, split=.8) -> None:
        self.root_dir = root_dir
        self.transform = transform

        self.path_images = os.path.join(self.root_dir, 'images')
        self.path_labels = os.path.join(self.root_dir, 'labels')

        self.image_paths = sorted(os.listdir(self.path_images))
        self.label_paths = sorted(os.listdir(self.path_labels))

        if train:
            self.image_paths = self.image_paths[:round(split*len(self.image_paths))]
            self.label_paths = self.label_paths[:round(split*len(self.label_paths))]
        else:
            self.image_paths = self.image_paths[round(split*len(self.image_paths)):]
            self.label_paths = self.label_paths[round(split*len(self.label_paths)):]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = io.imread(os.path.join(self.path_images, self.image_paths[index])) 
        label = io.imread(os.path.join(self.path_labels, self.label_paths[index])) 

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image.float(), label.float()


class RGBDataset(Dataset):
    def __init__(self, transform=None, train=True, split=.8):
        loaded = np.load("./data/training_data.npz")
        self.X = loaded["a"]
        self.y = loaded["b"]

        if train:
            self.images = self.X[:round(split*len(self.X))]
            self.labels = self.y[:round(split*len(self.y))]
        else:
            self.images = self.X[round(split*len(self.X)):]
            self.labels = self.y[round(split*len(self.y)):]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image.float(), label.float()


def load_data(dataset='isbi_em_seg', transformation=None, n_train=None, n_test=None, batch_size=2):

    ds_reg = [
        'isbi_em_seg',
        'isbi_em_seg_100',
        'rgb'
    ]

    if not dataset in ds_reg:
        print(f'Dataset not in registry, available datasets are: {ds_reg}')
    elif dataset == "rgb":
        transformation = ToTensor()
        train_set = RGBDataset(transform=transformation, train=True, split=0.8)  
        test_set = RGBDataset(transform=transformation, train=False, split=0.8)
        return DataLoader(train_set, batch_size=batch_size), DataLoader(test_set, batch_size=batch_size)
    else:   
        if not transformation:
            transformation = ToTensor()
        else:
            transformation = torch.nn.Sequential(
                ToTensor(),
                transformation
            )

        train_set = ISBIEMSegDataset(f'./data/{dataset}', transform=transformation, train=True, split=0.8)  
        test_set = ISBIEMSegDataset(f'./data/{dataset}', transform=transformation, train=False, split=0.8)

        return DataLoader(train_set, batch_size=batch_size), DataLoader(test_set, batch_size=batch_size)       

if __name__ == '__main__':
      
      train_set, test_set = load_data("rgb")
      print(len(train_set), len(test_set))