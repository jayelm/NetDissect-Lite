"""
Functions for returning masks
"""

import numpy as np
import settings
from tqdm import tqdm
from . import formula as F


def get_mask_global(masks, f):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    if isinstance(f, F.And):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        masks_both = []
        for ml, mr in zip(masks_l, masks_r):
            mb = mask_and(ml, mr)
            masks_both.append(mb)
        return masks_both
    elif isinstance(f, F.Or):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        masks_both = []
        for ml, mr in zip(masks_l, masks_r):
            mb = mask_or(ml, mr)
            masks_both.append(mb)
        return masks_both
    elif isinstance(f, F.Leaf):
        return masks[f.val]
    else:
        raise ValueError("Most be passed formula")


def mask_and(ml, mr):
    if ml is None:
        return None
    if mr is None:
        return None
    if len(ml.shape) == 0:
        # ML is 1 - and is mr
        return mr
    if len(mr.shape) == 0:
        return ml
    return np.bitwise_and(ml, mr)


def mask_or(ml, mr):
    if ml is None:
        return mr
    if mr is None:
        return ml
    if len(ml.shape) == 0:
        # ML is 1 - or is 1
        return ml
    if len(mr.shape) == 0:
        # MR is 1 - or is 1
        return mr
    return np.bitwise_or(ml, mr)


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
        data_size = self.prefetcher.segmentation.size()
        categories = self.prefetcher.segmentation.category_names()
        self.img2cat = np.zeros((data_size, len(categories)), dtype=np.uint8)
        if settings.PROBE_DATASET == 'broden':
            self.mask_shape = (112, 112)  # Hardcode this - hopefully it doesn't chnge
        else:
            self.mask_shape = (224, 224)

        n_batches = int(np.ceil(data_size / settings.TALLY_BATCH_SIZE))
        for batch in tqdm(self.prefetcher.batches(), desc='Loading masks',
                          total=n_batches):
            for concept_map in batch:
                img_index = concept_map['i']
                seg_shape = (concept_map['sh'], concept_map['sw'])
                for cat_i, cat in enumerate(categories):
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
                                self.initialize_mask(feat, 'pixel', seg_shape=seg_shape)
                            bin_mask = np.ones(seg_shape, dtype=np.bool)
                            if self.masks[feat][img_index] is None:
                                # Set as 1 to save space
                                self.masks[feat][img_index] = np.uint8(1)
                            elif len(self.masks[feat][img_index].shape) == 0:
                                # It's already a scalar label and we have another scalar label?
                                # Will this happen?
                                self.masks[feat][img_index] = np.uint8(1)
                            else:
                                # We have a mask and we're adding a scalar label.
                                # So override everywhere
                                print("Conflict")
                                self.masks[feat][img_index] = np.uint8(1)
                            # This image displays this category
                            self.img2cat[img_index][cat_i] = 1
                    else:
                        # Pixels
                        feats = np.unique(label_group.ravel())
                        for feat in feats:
                            # 0 is not a feature
                            if feat == 0:
                                continue
                            if feat not in self.masks:
                                self.initialize_mask(feat, 'pixel', seg_shape=seg_shape)
                            # We may want to make this COO sparse (let's look at memory usage)
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
                                    bin_mask = np.bitwise_or(bin_mask, (lg == feat))
                            if self.masks[feat][img_index] is None:
                                self.masks[feat][img_index] = bin_mask
                            elif len(self.masks[feat][img_index].shape) == 0:
                                # What does this mean - image was marked with
                                # e.g. category 86 as a scalar - now therer's
                                # also a mask - then which one do we care
                                # about?
                                print("Conflict")
                                pass
                            else:
                                # There are cases where features overlap across
                                # categories we've seen the feature before...not
                                # sure why (img_index 21/feature 86) but this is ignored in
                                # the original tally too so \shrug
                                self.masks[feat][img_index] = np.bitwise_or(self.masks[feat][img_index], bin_mask)
                            self.img2cat[img_index][cat_i] = 1

        self.labels = sorted(list(self.masks.keys()))

    def get_mask(self, f):
        return get_mask_global(self.masks, f)


    def initialize_mask(self, i, mask_type, seg_shape=(112, 112)):
        if i in self.masks:
            raise ValueError(f"Already initialized {i}")
        # Otherwise, this depends on whether i annotation is scalar or (I think)
        # NOTE: They may not be consistent, in which case change this code
        # NOTE: some features are both scalars and end up being other things too. that's fine
        #  if mask_type == 'scalar':
            #  self.masks[i] = np.zeros(self.prefetcher.segmentation.size(), dtype=np.uint8)
        if mask_type == 'pixel':
            self.masks[i] = [None for _ in range(self.prefetcher.segmentation.size())]
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
