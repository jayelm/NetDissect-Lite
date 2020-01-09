from functools import partial
import numpy
import os
import re
import random
import signal
import csv
import settings
import numpy as np
from collections import OrderedDict, defaultdict
from imageio import imread
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

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


def load_cub(data_dir, random_state=None, max_classes=None, train_only=False):
    class_names = sorted(os.listdir(os.path.join(data_dir, 'images')))
    c2i = dict((v, k) for k, v in enumerate(class_names))
    if max_classes is not None:
        class_names = class_names[:max_classes]
    class_imgs = {
        cn: np.load(os.path.join(data_dir, 'images', cn, 'img.npz'))
        for cn in class_names
    }
    # Load attributes
    index_fname = os.path.join(data_dir, 'images.txt')
    index = pd.read_csv(os.path.join(data_dir, 'images.txt'), names=['image_id', 'image_name'], sep=' ')
    name2id = dict(zip(index.image_name, index.image_id))
    id2name = {v: k for k, v in name2id.items()}
    attr_names_fname = os.path.join(data_dir, 'attributes', 'attributes.txt')
    attr_names = pd.read_csv(attr_names_fname, names=['attribute_id', 'attribute_name'],
                             sep=' ')
    attr_names['category'], attr_names['value'] = zip(*attr_names.attribute_name.str.split('::'))
    attr_names2id = dict(zip(attr_names.attribute_name, attr_names.attribute_id))
    id2attr_names = {v: k for k, v in attr_names2id.items()}

    attrs_fname = os.path.join(data_dir, 'attributes', 'image_attribute_labels.txt')
    attrs = pd.read_csv(attrs_fname, names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'],
                        sep=' ')
    id2attrs = defaultdict(dict)
    for img_id, img_name in id2name.items():
        img_attrs = attrs[attrs['image_id'] == img_id]
        # Should already be sorted, but double check
        assert img_attrs.shape[0] == 312
        img_attrs = img_attrs.sort_values('attribute_id')['is_present']
        id2attrs[img_name] = img_attrs.to_numpy().astype(np.uint8)

    # Flatten
    imgs = []
    classes = []
    img_ids = []
    attrs = []
    for c, cimgs in tqdm(class_imgs.items(), desc='Loading CUB classes'):
        for img_name in cimgs.keys():
            imgs.append(cimgs[img_name])
            classes.append(c2i[c])
            img_ids.append(name2id[img_name])
            attrs.append(id2attrs[img_name])

    tloader = TransformLoader(224)
    train_transform = tloader.get_composed_transform(True)
    if train_only:
        train = [imgs, classes, attrs]
        return CUBDataset(train, class_names, transform=train_transform,
                          attr_metadata=attr_names)
    else:
        test_transform = tloader.get_composed_transform(False)
        train, val, test = train_val_test_split(imgs, classes, attrs,
                                                val_size=0.15, test_size=0.15,
                                                random_state=random_state)
        return {
            'train': CUBDataset(train, class_names, transform=train_transform,
                                attr_metadata=attr_names),
            'val': CUBDataset(val, class_names, transform=test_transform,
                              attr_metadata=attr_names),
            'test': CUBDataset(test, class_names, transform=test_transform,
                               attr_metadata=attr_names),
        }


class CUBDataset:
    def __init__(self, tensors, class_names, transform=None, attr_metadata=None):
        self.imgs, self.classes, self.attrs = tensors
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.transform = transform
        self.attr_metadata = attr_metadata

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform is not None:
            img = self.transform(img)
        attrs = self.attrs[i]
        c = self.classes[i]
        return img, c, attrs

    def __len__(self):
        return len(self.imgs)


class CUBPrefetcher:
    '''
    Load CUB images and attributes.
    This is mostly just a wrapper to satisfy the prefetcher API.
    '''
    def __init__(self, data, split=None, randomize=False,
            segmentation_shape=None, categories=None, once=False,
            start=None, end=None, batch_size=4, ahead=4, thread=False):
        '''
        Constructor arguments:
        data: The CUBDataset to load.
        split: None for no filtering, or 'train' or 'val' etc.
        randomize: True to randomly shuffle order, or a random seed.
        categories: a list of categories to include in each batch.
        batch_size: number of data items for each batch.
        ahead: the number of data items to prefetch ahead.
        '''
        self.cub = data
        self.batch_size = batch_size
        self.cub_loader = to_dataloader(self.cub, batch_size=batch_size)
        self.indexes = range(0, len(self.cub))

    def tensor_batches(self, bgr_mean=None):
        return self.cub_loader

    def __len__(self):
        return len(self.cub_loader)


def to_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, pin_memory=True)
