"""
Functions for returning masks
"""

import numpy as np
import settings
from tqdm import tqdm
from . import formula as F

import os
import pickle
from pycocotools import mask as cmask


ONES = cmask.encode(np.ones((112, 112), dtype=np.bool, order='F'))


def get_mask_global(masks, f):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # TODO: Handle here when doing AND and ORs of scenes vs scalars.
    if isinstance(f, F.And):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        l_nd = isinstance(masks_l, np.ndarray)
        r_nd = isinstance(masks_r, np.ndarray)
        if l_nd and r_nd:
            return np.logical_and(masks_l, masks_r)
        elif l_nd:
            res = []
            for ml, mr in zip(masks_l, masks_r):
                if not ml:
                    # Left is 0 everywhere - therefore 0 conj
                    res.append(None)
                else:
                    res.append(mr)
            return res
        elif r_nd:
            res = []
            for ml, mr in zip(masks_l, masks_r):
                if not mr:
                    res.append(None)
                else:
                    res.append(ml)
            return res
        else:
            res = []
            for ml, mr in zip(masks_l, masks_r):
                if ml is None or mr is None:
                    res.append(None)
                else:
                    res.append(cmask.merge((ml, mr), intersect=True))
            return res
    elif isinstance(f, F.Or):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        l_nd = isinstance(masks_l, np.ndarray)
        r_nd = isinstance(masks_r, np.ndarray)
        if l_nd and r_nd:
            return np.logical_or(masks_l, masks_r)
        elif l_nd:
            res = []
            for ml, mr in zip(masks_l, masks_r):
                if ml:
                    # Left is 1 everywhere - therefore 1 conj
                    res.append(ONES)
                else:
                    res.append(mr)
            return res
        elif r_nd:
            res = []
            for ml, mr in zip(masks_l, masks_r):
                if mr:
                    res.append(ONES)
                else:
                    res.append(ml)
            return res
        else:
            res = []
            for ml, mr in zip(masks_l, masks_r):
                if ml is None or mr is None:
                    res.append(None)
                else:
                    res.append(cmask.merge((ml, mr), intersect=False))
            return res
    elif isinstance(f, F.Not):
        masks_val = get_mask_global(masks, f.val)
        if isinstance(masks_val, np.ndarray):
            return np.logical_not(masks_val)
        else:
            new_masks_val = []
            for m in masks_val:
                if m is None:
                    # TODO: Use TRUE here for efficiency?
                    new_masks_val.append(ONES)
                else:
                    inv = cmask.invert(m)
                    new_masks_val.append(inv)
            return new_masks_val
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
        # initialize the masks dictionary.
        # For scalars - this is simply a binary np array - 1 if the scalar applies or not
        # For masks - this is a regular python list of Option[np.arrays] -
        # where each np array is a binary array (or maybe a sparse matrix tbd)
        # We need to know whether features are scalars or not
        # Note that this depends on the specific feature and not the (weirdly).
        #  self.cache_file = os.path.join(settings.DATA_DIRECTORY, 'masks.pkl')
        #  if os.path.exists(self.cache_file):
            #  with open(self.cache_file) as f:
                #  self.masks = pickle.load(f)
        self.masks = {}
        self.data_size = self.prefetcher.segmentation.size()
        self.categories = self.prefetcher.segmentation.category_names()
        self.n_labels = len(self.prefetcher.segmentation.primary_categories_per_index())
        self.img2cat = np.zeros((self.data_size, len(self.categories)), dtype=np.bool)
        self.img2label = np.zeros((self.data_size, self.n_labels), dtype=np.bool)
        if settings.PROBE_DATASET == 'broden':
            self.mask_shape = (112, 112)  # Hardcode this - hopefully it doesn't chnge
        else:
            self.mask_shape = (224, 224)

        rle_masks_file = os.path.join(settings.DATA_DIRECTORY, f"rle_masks{'_test' if settings.TEST_MODE else ''}.pkl")
        if os.path.exists(rle_masks_file):
            with open(rle_masks_file, 'rb') as f:
                cache = pickle.load(f)
                self.masks = cache['masks']
                self.img2cat = cache['img2cat']
                self.img2label = cache['img2label']
        else:
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
                                    # Treat it at pixel-level because a feature can be bgoth a pixel-level annotation and a  may
                                    # FIXME: Actually, is that why the cat information is there?
                                    # But tally labels doesn't care abouut categories...so are they just treated like overlaps?
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
                                # parts belonging to differrent objects. afaict
                                # these are ignored during normal tallying
                                # (np.concatenate followed by np.bincount.ravel())
                                if label_group.shape[0] == 1:
                                    bin_mask = (label_group.squeeze(0) == feat)
                                else:
                                    bin_mask = np.zeros_like(label_group[0])
                                    for lg in label_group:
                                        bin_mask = np.logical_or(bin_mask, (lg == feat))
                                if isinstance(self.masks[feat], np.ndarray):
                                    # Sometimes annotation is both pixel and scalar level. Retroactively fix this
                                    print(f"Coercing {feat} to pixel level")
                                    self.masks[feat] = [(None if i == 0 else np.ones(*self.mask_shape, dtype=np.bool)) for i in self.masks[feat]]
                                if self.masks[feat][img_index] is None:
                                    self.masks[feat][img_index] = bin_mask
                                else:
                                    self.masks[feat][img_index] = np.logical_or(self.masks[feat][img_index], bin_mask)
                                self.img2label[img_index, feat] = True
                                self.img2cat[img_index][cat_i] = True

            # Convert pixel-level masks to RLE encoding
            for feat, mask in tqdm(self.masks.items(), total=len(self.masks), desc='RLE'):
                if isinstance(mask, list):
                    new_masks = []
                    for m in mask:
                        if m is None:
                            new_masks.append(None)
                        else:
                            m = np.asfortranarray(m)
                            new_masks.append(cmask.encode(m))
                    self.masks[feat] = new_masks
            with open(rle_masks_file, 'wb') as f:
                pickle.dump({
                    'masks': self.masks,
                    'img2label': self.img2label,
                    'img2cat': self.img2cat
                }, f)

        self.labels = sorted(list(self.masks.keys()))

    def get_mask(self, f):
        return get_mask_global(self.masks, f)


    def initialize_mask(self, i, mask_type):
        if i in self.masks:
            raise ValueError(f"Already initialized {i}")
        # Otherwise, this depends on whether i annotation is scalar or (I think)
        # NOTE: They may not be consistent, in which case change this code
        # NOTE: some features are both scalars and end up being other things too. that's fine
        if mask_type == 'scalar':
            self.masks[i] = np.zeros(self.data_size, dtype=np.bool)
        elif mask_type == 'pixel':
            self.masks[i] = [None for _ in range(self.data_size)]
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
