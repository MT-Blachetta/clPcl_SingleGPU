from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from torch.utils.data import Dataset
from utils.mypath import MyPath
import os
import numpy as np
import torchvision.datasets

class MNIST(Dataset):

    def __init__(self, root=MyPath.db_root_dir('mnist'), split='train', transform=None,download=False):

        super(MNIST, self).__init__()
        self.root = root
        self.transform = transform
        self.classes = ['0','1','2','3','4','5','6','7','8','9']
        #self.split = verify_str_arg(split, "split", self.splits)
        
        if split == 'train':
            self.dataset = torchvision.datasets.MNIST(root, train=True, transform=None, target_transform= None, download=download)
        else:
            self.dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform= None, download=download)
            
        #if self.transform is not None:
            
    def __getitem__(self, index):

        ds = self.dataset[index]
        img = ds[0]
        target = ds[1]
        img_size = img.size
        class_name = str(target)
        
        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        return out
        
    def get_image(self, index):
        img = self.dataset[index][0]
        
        return img

    def __len__(self):
        return len(self.dataset)
