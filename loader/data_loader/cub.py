from functools import partial
import numpy
import os
import re
import random
import signal
import csv
import settings
import numpy as np
from collections import OrderedDict
from imageio import imread
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from tqdm import tqdm

from PIL import ImageEnhance


transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color)


def train_val_test_split(*args,
                         val_size=0.1,
                         test_size=0.1,
                         random_state=None,
                         **kwargs):
    """
    Split data into train, validation, and test splits
    """
    assert [len(a) == len(args[0]) for a in args], "Uneven lengths"
    idx = np.arange(len(args[0]))
    idx_train, idx_valtest = train_test_split(idx,
                                              test_size=val_size + test_size,
                                              random_state=random_state,
                                              shuffle=True)
    idx_val, idx_test = train_test_split(idx_valtest,
                                         test_size=test_size /
                                         (val_size + test_size),
                                         random_state=random_state,
                                         shuffle=True)
    train = [[a[i] for i in idx_train] for a in args]
    val = [[a[i] for i in idx_val] for a in args]
    test = [[a[i] for i in idx_test] for a in args]
    return train, val, test


class ImageJitter:
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k])
                           for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


class TransformLoader:
    def __init__(self,
                 image_size,
                 normalize_param=dict(
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method(
                [int(self.image_size * 1.15),
                 int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False, normalize=True, to_pil=True):
        if aug:
            transform_list = [
                'RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip',
                'ToTensor'
            ]
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor']

        if normalize:
            transform_list.append('Normalize')

        if to_pil:
            transform_list = ['ToPILImage'] + transform_list

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_normalize(self):
        return self.parse_transform('Normalize')


def load_cub(data_dir, random_state=None, max_classes=None):
    class_names = sorted(os.listdir(os.path.join(data_dir, 'images')))
    c2i = dict((v, k) for k, v in enumerate(class_names))
    if max_classes is not None:
        class_names = class_names[:max_classes]
    class_imgs = {
        cn: np.load(os.path.join(data_dir, 'images', cn, 'img.npz'))
        for cn in class_names
    }
    # Flatten
    imgs = []
    classes = []
    for c, cimgs in tqdm(class_imgs.items(), desc='Loading CUB classes'):
        for k in cimgs.keys():
            imgs.append(cimgs[k])
            classes.append(c2i[c])

    train, val, test = train_val_test_split(imgs, classes, val_size=0.15, test_size=0.15,
                                            random_state=random_state)
    tloader = TransformLoader(224)
    train_transform = tloader.get_composed_transform(True)
    test_transform = tloader.get_composed_transform(False)
    return {
        'train': CUBDataset(train, class_names, transform=train_transform),
        'val': CUBDataset(val, class_names, transform=test_transform),
        'test': CUBDataset(test, class_names, transform=test_transform),
    }


class CUBDataset:
    def __init__(self, tensors, class_names, transform=None):
        self.imgs, self.classes = tensors
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.transform = transform

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform is not None:
            img = self.transform(img)
        c = self.classes[i]
        return img, c

    def __len__(self):
        return len(self.imgs)
