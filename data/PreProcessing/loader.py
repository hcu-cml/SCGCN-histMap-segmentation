from data.PreProcessing import *
from data.PreProcessing.augmt import *

import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms

from data.PreProcessing.pre_process import *


class AlgricultureDataset(Dataset):
    def __init__(self, mode='train', file_lists=None, windSize=(256, 256),
                 num_samples=300, pre_norm=False, scale=1.0 / 1.0, augment = False):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.norm = pre_norm
        self.winsize = windSize
        self.samples = num_samples
        self.scale = scale
        self.all_ids = file_lists['all_files']
        self.image_files = file_lists[IMG]  # image_files = [[bands1, bands2,..], ...]
        self.mask_files = file_lists[GT]    # mask_files = [gt1, gt2, ...]
        self.augmentation = augment

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):

        filename = self.image_files[idx]
        image = imload(filename, scale_rate=self.scale)
        label = imload(self.mask_files[idx], gray=True, scale_rate=self.scale)

        if self.winsize != np.shape(label):
            image, label = img_mask_crop(image=image, mask=label,
                                         size=self.winsize, limits=self.winsize)

        if self.augmentation:
            if self.mode == 'train':
                image_p, label_p = self.train_augmentation(image, label)
            else:
                image_p, label_p = self.val_augmentation(image, label)
        else:
            image_p = image
            label_p = label

        image_p = np.asarray(image_p, np.float32).transpose((2, 0, 1)) / 255.0
        label_p = np.asarray(label_p, dtype='int64')

        image_p, label_p = torch.from_numpy(image_p), torch.from_numpy(label_p)

        if self.norm:
            image_p = self.normalize(image_p)

        return image_p, label_p

    @classmethod
    def train_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def val_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def normalize(cls, img):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        norm = standard_transforms.Compose([standard_transforms.Normalize(*mean_std)])
        return norm(img)

