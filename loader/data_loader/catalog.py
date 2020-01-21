"""
Functions for returning masks
"""

import numpy as np
import settings
from tqdm import tqdm
from . import formula as F
import torch


def to_dense(f):
    def to_dense_wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        return res.todense()
    return to_dense_wrapper


def get_mask_global(masks, f):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # TODO: Handle here when doing AND and ORs of scenes vs scalars.
    if isinstance(f, F.And):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        if masks_l.ndim == 1 and masks_r.ndim == 3:
            masks_l = masks_l.unsqueeze(1).unsqueeze(1)
        elif masks_l.ndim == 3 and masks_r.ndim == 1:
            masks_r = masks_r.unsqueeze(1).unsqueeze(1)
        return masks_l & masks_r
    elif isinstance(f, F.Or):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        if masks_l.ndim == 1 and masks_r.ndim == 3:
            masks_l = masks_l.unsqueeze(1).unsqueeze(1)
        elif masks_l.ndim == 3 and masks_r.ndim == 1:
            masks_r = masks_r.unsqueeze(1).unsqueeze(1)
        return masks_l | masks_r
    elif isinstance(f, F.Not):
        masks_val = get_mask_global(masks, f.val)
        return ~masks_val
    elif isinstance(f, F.Leaf):
        return masks[f.val]
    else:
        raise ValueError("Most be passed formula")


class MaskCatalog:
    # A map from
    # label -> [list of Option[np.arrays]]
    def __init__(self, prefetcher, cache=True):
        # Loop through prefetcher batches, collecting
        self.prefetcher = prefetcher
        self.masks = {}
        self.data_size = self.prefetcher.segmentation.size()
        self.categories = self.prefetcher.segmentation.category_names()
        self.n_labels = len(self.prefetcher.segmentation.primary_categories_per_index())
        self.img2cat = np.zeros((self.data_size, len(self.categories)), dtype=np.bool)
        self.img2label = np.zeros((self.data_size, self.n_labels), dtype=np.bool)
        if settings.PROBE_DATASET == 'broden':
            self.mask_shape = (112, 112)
        else:
            self.mask_shape = (224, 224)

        n_batches = int(np.ceil(self.data_size / settings.TALLY_BATCH_SIZE))
        for batch in tqdm(self.prefetcher.batches(), desc='Loading masks',
                          total=n_batches):
            for concept_map in batch:
                img_index = concept_map['i']
                for cat_i, cat in enumerate(self.categories):
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        # Scalar
                        for feat in label_group:
                            if feat == 0:
                                # Somehow 0 is a feature?
                                # Just continue - it exists in index.csv under scenes but not in c_scenes
                                continue
                            if feat not in self.masks:
                                self.initialize_mask(feat, 'scalar')
                            self.masks[feat][img_index] = True
                            # This image displays this category
                            self.img2cat[img_index][cat_i] = True
                            self.img2label[img_index, feat] = True
                    else:
                        # Pixels
                        feats = np.unique(label_group.ravel())
                        for feat in feats:
                            # 0 is not a feature
                            if feat == 0:
                                continue
                            if feat not in self.masks:
                                self.initialize_mask(feat, 'pixel')
                            # NOTE: sometimes label group is > 1 length (e.g.
                            # for parts) which means there are overlapping
                            # parts belonging to different objects. afaict
                            # these are ignored during normal tallying
                            # (np.concatenate followed by np.bincount.ravel())
                            if label_group.shape[0] == 1:
                                bin_mask = (label_group.squeeze(0) == feat)
                            else:
                                bin_mask = np.zeros_like(label_group[0])
                                for lg in label_group:
                                    bin_mask = np.logical_or(bin_mask, (lg == feat))
                            try:
                                self.masks[feat][img_index] = np.logical_or(self.masks[feat][img_index], bin_mask)
                            except ValueError:
                                print(f"Coercing {feat} to mask level")
                                # Overlaps where a feature is mask AND
                                # image-level. Retroactively coerce to mask
                                # level
                                expanded = np.tile(self.masks[feat][:, np.newaxis, np.newaxis], (1, *self.mask_shape))
                                self.masks[feat] = expanded
                                self.masks[feat][img_index] = np.logical_or(self.masks[feat][img_index], bin_mask)
                            self.img2label[img_index, feat] = True
                            self.img2cat[img_index, cat_i] = True
        # To torch
        self.mask_tensors = {k: torch.from_numpy(v) for k, v in self.masks.items()}
        # To sparse
        self.img2label = torch.from_numpy(self.img2label)
        self.img2cat = torch.from_numpy(self.img2cat)

        self.labels = sorted(list(self.masks.keys()))

    def get_mask(self, f):
        return get_mask_global(self.masks, f)

    def initialize_mask(self, i, mask_type):
        if i in self.masks:
            raise ValueError(f"Already initialized {i}")
        if mask_type == 'scalar':
            self.masks[i] = np.zeros(self.data_size, dtype=np.bool)
        elif mask_type == 'pixel':
            self.masks[i] = np.zeros((self.data_size, *self.mask_shape), dtype=np.bool)
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
