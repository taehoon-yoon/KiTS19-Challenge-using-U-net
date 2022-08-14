import torch
import os
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms


class makeDataset(Dataset):
    def __init__(self, kind, location, tuning=False, tune_size=0.1, transform=None, normalize=False):
        '''
        kind must be either 'train' or 'valid'
        location must be either 'data_npy' or 'data_npy_filterd' or 'data_npy_2ch
        '''
        self.kind = kind
        self.root = os.path.join('C:\\', 'Users', 'willy', 'Desktop', 'kits', location, kind)
        self.img_path = os.path.join(self.root, 'image')
        self.seg_path = os.path.join(self.root, 'segmentation')
        self.tuning = tuning
        self.total = 37250 if self.kind == 'train' else 7922
        if location == 'data_npy_filterd':
            self.total = 14086
        self.tune_list = []
        self.tune_cnt = 0
        self.transform = transform
        self.normalization = normalize
        if self.normalization:
            self.normalize = transforms.Normalize((0.5), (0.5))

        if self.tuning:
            self.tune_cnt = int(self.total * tune_size)
            self.tune_list = np.random.choice(self.total, self.tune_cnt, replace=False)

    def __len__(self):
        num = self.tune_cnt if self.tuning else self.total
        return num

    def __getitem__(self, idx):
        idx = self.tune_list[idx] if self.tuning else idx

        img_id = "{:05d}.npy".format(idx)
        final_img_path = os.path.join(self.img_path, img_id.format(idx))
        final_seg_path = os.path.join(self.seg_path, img_id.format(idx))
        img = torch.tensor(np.load(final_img_path), dtype=torch.float32)
        seg = torch.tensor(np.load(final_seg_path), dtype=torch.uint8)
        if self.normalization:
            img = self.normalize(img)

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)

            random.seed(seed)
            torch.manual_seed(seed)
            seg = self.transform(seg)

        return img, seg
