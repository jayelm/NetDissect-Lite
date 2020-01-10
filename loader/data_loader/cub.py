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
from torch.nn.functional import one_hot
import pandas as pd
from PIL import Image
from torchvision.transforms import CenterCrop

from . import data_utils as du

from PIL import ImageEnhance


PARTS = [
    'back',
    'beak',
    'belly',
    'breast',
    'crown',
    'forehead',
    'left eye',
    'left leg',
    'left wing',
    'nape',
    'right eye',
    'right leg',
    'right wing',
    'tail',
    'throat',
]


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


def load_cub(data_dir, random_state=None, max_classes=None, train_only=False,
             train_augment=True):
    """
    Load CUB dataset.

    WE REDUCE image and attribute ids by 1 here so we never have to worry about
    off-by-one errors anywhere else.
    """
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
    index['image_id'] = index['image_id'] - 1
    name2id = dict(zip(index.image_name, index.image_id))
    id2name = {v: k for k, v in name2id.items()}
    attr_names_fname = os.path.join(data_dir, 'attributes', 'attributes.txt')
    attr_names = pd.read_csv(attr_names_fname, names=['attribute_id', 'attribute_name'],
                             sep=' ')
    attr_names['attribute_id'] = attr_names['attribute_id'] - 1
    attr_names['category'], attr_names['value'] = zip(*attr_names.attribute_name.str.split('::'))
    #  attr_names2id = dict(zip(attr_names.attribute_name, attr_names.attribute_id))
    #  id2attr_names = {v: k for k, v in attr_names2id.items()}

    attrs_fname = os.path.join(data_dir, 'attributes', 'image_attribute_labels.txt')
    id2attrs = defaultdict(list)
    with open(attrs_fname, 'r') as f:
        for line in tqdm(f, desc='Loading attributes', total=3677856):
            image_id, _, is_present, *_ = line.split()
            id2attrs[int(image_id) - 1].append(int(is_present))
    id2attrs = {k: np.array(v, dtype=np.uint8) for k, v in id2attrs.items()}

    # Load images in ascending image id order
    imgs = []
    classes = []
    img_ids = []
    attrs = []
    for image_id, image_name in tqdm(zip(index.image_id, index.image_name),
                                     desc='Loading CUB images', total=len(index.image_id)):
        image_cl = image_name.split('/')[0]
        if image_cl not in class_imgs and max_classes is not None:
            continue
        imgs.append(class_imgs[image_cl][image_name])
        classes.append(c2i[image_cl])
        img_ids.append(image_id)
        attrs.append(id2attrs[image_id])
    assert all(i == j for i, j in zip(img_ids, range(len(classes))))

    tloader = TransformLoader(224)
    train_transform = tloader.get_composed_transform(train_augment)
    if train_only:
        train = [imgs, classes, attrs]
        return CUBDataset(train, class_names, transform=train_transform,
                          attr_metadata=attr_names, image_metadata=index)
    else:
        test_transform = tloader.get_composed_transform(False)
        train, val, test = train_val_test_split(imgs, classes, attrs,
                                                val_size=0.15, test_size=0.15,
                                                random_state=random_state)
        return {
            'train': CUBDataset(train, class_names, transform=train_transform,
                                attr_metadata=attr_names, image_metadata=index),
            'val': CUBDataset(val, class_names, transform=test_transform,
                              attr_metadata=attr_names, image_metadata=index),
            'test': CUBDataset(test, class_names, transform=test_transform,
                               attr_metadata=attr_names, image_metadata=index),
        }


class CUBDataset:
    def __init__(self, tensors, class_names, transform=None, attr_metadata=None,
                 image_metadata=None):
        self.imgs, self.classes, self.attrs = tensors
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.transform = transform
        self.attr_metadata = attr_metadata
        self.image_metadata = image_metadata
        self.attr_categories = list(self.attr_metadata['category'].unique())
        self.attr_names = list(self.attr_metadata['attribute_name'])
        self.attr2cat = dict(zip(self.attr_metadata['attribute_id'],
                                 self.attr_metadata['category']))
        self.cat2id = dict((v, k) for k, v in enumerate(self.attr_metadata['category'].unique()))
        # From attr ids to category ids
        self.labelcat = du.onehot(self.primary_categories_per_index())

    def primary_categories_per_index(self):
        # ATTRS
        #  return np.array([self.cat2id[cat] for cat in self.attr_metadata['category']])
        return np.zeros(len(PARTS), dtype=np.int)

    def category_names(self):
        # ATTRS
        #  return self.attr_categories
        return ['part']

    def name(self, category, j):
        # ATTRS
        #  if category is not None:
            #  raise NotImplementedError
        #  return self.attr_names[j]
        return PARTS[j]

    @property
    def label(self):
        # ATTRS
        #  return self.attr_names
        return PARTS

    def attr_to_cm(self, attr, idx):
        is_present = np.argwhere(attr).squeeze()
        cm = {cat: [] for cat in self.attr_categories}
        for i in is_present:
            i_cat = self.attr2cat[i]
            cm[i_cat].append(i)
        cm['i'] = idx
        # Assume FULL segmentation mask
        cm['sh'] = 224
        cm['sw'] = 224
        return cm

    def filename(self, i):
        fname = self.image_metadata.loc[i, 'image_name']
        fname = os.path.join(settings.DATA_DIRECTORY, 'images', fname)
        return fname

    def to_concept_map(self, attrs, ids):
        return [self.attr_to_cm(a, i) for a, i in zip(attrs, ids)]

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform is not None:
            img = self.transform(img)
        attrs = self.attrs[i]
        c = self.classes[i]
        return img, c, i, attrs

    def __len__(self):
        return len(self.imgs)

    def size(self):
        """Per data-loader API this is the size of the underlying dataset"""
        return len(self)


class CUBSegmentationPrefetcher:
    '''
    Load CUB images and attributes.
    This is mostly just a wrapper to satisfy the prefetcher API.
    Loads segmentations too
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
        self.cub_loader = to_dataloader(self.cub, batch_size=batch_size,
                                        shuffle=randomize)
        self.indexes = range(0, len(self.cub))
        self.segmentations = {
            cn: np.load(os.path.join(settings.DATA_DIRECTORY, 'parts', 'segmentations', f'{cn}.npz'))
            for cn in data.class_names
        }
        self.seg_transformer = CenterCrop(size=(244, 244))

    def categories(self):
        return self.cub.category_names()

    @property
    def label(self):
        return self.cub.label

    def batches(self):
        for *_, ids, _ in self.cub_loader:
            cm = self.fetch_segmentations(ids.numpy())
            yield cm

    def fetch_segmentations(self, ids):
        cms = []
        for i in ids:
            cm = {}
            name = self.cub.image_metadata.loc[i, 'image_name']
            bc, img_name = name.split('/')
            cm['fn'] = name
            cm['i'] = i
            # Just one category for now
            seg = self.segmentations[bc][name]
            seg = Image.fromarray(seg).resize((257, 257), resample=Image.NEAREST)
            seg = np.array(self.seg_transformer(seg))
            cm['part'] = seg
            cm['sh'] = 224
            cm['sw'] = 224
            cms.append(cm)
        return cms

    # ATTRS
    #  def batches(self):
        #  # Get concept map
        #  # Assume FULL segmentation height here.
        #  for *_, ids, attrs in self.cub_loader:
            #  attrs = attrs.numpy()
            #  cm = self.cub.to_concept_map(attrs, ids)
            #  yield cm

    def tensor_batches(self, bgr_mean=None):
        return self.cub_loader

    def __len__(self):
        return len(self.cub_loader)


def to_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True)
