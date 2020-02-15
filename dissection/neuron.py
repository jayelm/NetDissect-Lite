import os
from PIL import Image
import numpy as np
import torch
import settings
import util.upsample as upsample
import pandas as pd
import util.vecquantile as vecquantile
from util.misc import safe_layername
import multiprocessing.pool as pool
import multiprocessing as mp
from loader.data_loader.broden import load_csv
from loader.data_loader.broden import SegmentationData, SegmentationPrefetcher
from loader.data_loader.catalog import MaskCatalog, get_mask_global
from loader.data_loader import formula as F
from loader.data_loader.cub import load_cub, CUBSegmentationPrefetcher
from loader.data_loader import gqa
from tqdm import tqdm, trange
import csv
from collections import Counter
import itertools

from pycocotools import mask as cmask

# Janky - globals for multiprocessing to prevent shared memory
g = {}

features_blobs = []
def hook_feature(module, inp, output):
    features_blobs.append(output.data.cpu().numpy())


def upsample_features(features, shape):
    return np.array(Image.fromarray(features).resize(shape, resample=Image.BILINEAR))


class NeuronOperator:
    def __init__(self):
        os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'image'), exist_ok=True)
        if settings.PROBE_DATASET == 'broden':
            self.data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATAGORIES)
            self.loader = SegmentationPrefetcher(self.data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
            self.mean = [109.5388,118.6897,124.6901]
        elif settings.PROBE_DATASET == 'cub':
            self.data = load_cub(settings.DATA_DIRECTORY, train_only=True,
                                 max_classes=5 if settings.TEST_MODE else None,
                                 train_augment=False)
            self.loader = CUBSegmentationPrefetcher(self.data, once=True, batch_size=settings.BATCH_SIZE)
            self.mean = None
        elif settings.PROBE_DATASET == 'gqa':
            self.data = gqa.load_gqa(settings.DATA_DIRECTORY,
                                     max_images=5 if settings.TEST_MODE else None)

            #  TODO: GQA. May want to return batches separate of
            #  images and a function for getting the right id
            self.loader = self.data
            self.mean = None


    def feature_extraction(self, model=None, memmap=True):
        loader = self.loader
        # extract the max value activaiton for each image
        maxfeatures = [None] * len(settings.FEATURE_NAMES)
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        # FIXME: Multiple files for preds/logits is completely redundant
        all_preds = [None] * len(settings.FEATURE_NAMES)
        all_logits = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files =  [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % safe_layername(feature_name))  for feature_name in  settings.FEATURE_NAMES]
            mmap_max_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_max.mmap" % safe_layername(feature_name)) for feature_name in settings.FEATURE_NAMES]
            mmap_pred_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_pred.mmap" % safe_layername(feature_name)) for feature_name in settings.FEATURE_NAMES]
            mmap_logit_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_logit.mmap" % safe_layername(feature_name)) for feature_name in settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file, mmap_pred_file, mmap_logit_file) in enumerate(zip(mmap_files, mmap_max_files, mmap_pred_files, mmap_logit_files)):
                if os.path.exists(mmap_file) and os.path.exists(mmap_max_file) and os.path.exists(mmap_pred_file) and os.path.exists(mmap_logit_file) and features_size[i] is not None:
                    print('loading features %s' % safe_layername(settings.FEATURE_NAMES[i]))
                    wholefeatures[i] = np.memmap(mmap_file, dtype=np.float32, mode='r', shape=tuple(features_size[i]))
                    maxfeatures[i] = np.memmap(mmap_max_file, dtype=np.float32, mode='r', shape=tuple(features_size[i][:2]))
                    all_preds[i] = np.memmap(mmap_pred_file, dtype=np.int64, mode='r', shape=(features_size[i][0], 2))
                    all_logits[i] = np.memmap(mmap_logit_file, dtype=np.float32, mode='r', shape=(features_size[i][0], settings.NUM_CLASSES))
                else:
                    print('file missing, loading from scratch')
                    skip = False
            if skip:
                return wholefeatures, maxfeatures, all_preds, all_logits

        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        for batch_idx, (inp, *rest) in tqdm(enumerate(loader.tensor_batches(bgr_mean=self.mean, global_labels=True)), desc='Extracting features', total=int(np.ceil(num_batches))):
            del features_blobs[:]
            if settings.PROBE_DATASET == 'broden':
                # Unused for CUB - tensors already
                inp = torch.from_numpy(inp[:, ::-1, :, :].copy())
                inp.div_(255.0 * 0.224)
            if settings.GPU:
                inp = inp.cuda()
            with torch.no_grad():
                logits = model.forward(inp)

            while np.isnan(logits.data.cpu().max()):
                print("nan")
                del features_blobs[:]
                logits = model.forward(inp)

            preds = logits.argmax(1).cpu()
            if settings.PROBE_DATASET == 'cub':
                # Targets are provided
                targets = rest[0]
            elif settings.PROBE_DATASET == 'broden':
                targets = rest[0].squeeze(1)
            else:
                # Model was not trained to predict
                targets = preds

            if all_preds[0] is None:
                for i, feat_batch in enumerate(features_blobs):
                    size_preds = (len(loader.indexes), 2)
                    if memmap:
                        all_preds[i] = np.memmap(mmap_pred_files[i], dtype=np.int64, mode='w+', shape=size_preds)
                    else:
                        all_preds[i] = np.zeros(size_preds, dtype=np.int64)

            if all_logits[0] is None:
                for i, feat_batch in enumerate(features_blobs):
                    size_logits = (len(loader.indexes), settings.NUM_CLASSES)
                    if memmap:
                        all_logits[i] = np.memmap(mmap_logit_files[i], dtype=np.float32, mode='w+', shape=size_logits)
                    else:
                        all_logits[i] = np.zeros(size_logits, dtype=np.float32)

            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    if memmap:
                        maxfeatures[i] = np.memmap(mmap_max_files[i],dtype=np.float32,mode='w+',shape=size_features)
                    else:
                        maxfeatures[i] = np.zeros(size_features)

            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (
                    len(loader.indexes), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i], dtype=np.float32, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)

            np.save(features_size_file, features_size)
            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blobs):
                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch
                all_preds[i][start_idx:end_idx] = np.stack((preds, targets), 1)
                all_logits[i][start_idx:end_idx] = logits.cpu().numpy()
        if len(feat_batch.shape) == 2:
            wholefeatures = maxfeatures
        return wholefeatures, maxfeatures, all_preds, all_logits

    def quantile_threshold(self, features, savepath=''):
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
            print(f"Loading cached quantiles {qtpath}")
            return np.load(qtpath)
        print("calculating quantile threshold")
        quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
        batch_size = 64
        for i in trange(0, features.shape[0], batch_size, desc='Processing quantiles'):
            batch = features[i:i + batch_size]
            batch = np.transpose(batch, axes=(0, 2, 3, 1)).reshape(-1, features.shape[1])
            quant.add(batch)
        ret = quant.readout(1000)[:, int(1000 * (1-settings.QUANTILE)-1)]
        if savepath:
            np.save(qtpath, ret)
        return ret
        # return np.percentile(features,100*(1 - settings.QUANTILE),axis=axis)

    @staticmethod
    def tally_job(args):
        return NeuronOperator.tally_job_search(args)

    @staticmethod
    def get_uhits(args):
        u, = args
        ufeat = g['features'][:, u]
        uthresh = g['threshold'][u]
        mask_shape = g['mask_shape']
        uidx = np.argwhere(ufeat.max((1, 2)) > uthresh).squeeze(1)
        ufeat = np.array([upsample_features(ufeat[i], mask_shape) for i in uidx])

        # Create full array
        uhitidx = np.zeros((g['features'].shape[0], *mask_shape), dtype=np.bool)

        # Get indices where threshold is exceeded
        uhit_subset = ufeat > uthresh
        uhitidx[uidx] = uhit_subset

        # Get lengths of those indicees
        uhits = uhit_subset.sum()

        # Save as compressed
        uhitidx_flat = uhitidx.reshape((uhitidx.shape[0] * uhitidx.shape[1], uhitidx.shape[2]))
        uhit_mask = cmask.encode(np.asfortranarray(uhitidx_flat))

        return u, uidx, uhit_mask, uhits

    @staticmethod
    def tally_job_search(args):
        features, data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, start, end, savepath, csvpath = args

        categories = data.category_names()
        pcpi = data.primary_categories_per_index()
        g['pcpi'] = pcpi
        pcats = data.primary_categories_per_index()
        units = features.shape[1]

        if settings.PROBE_DATASET == 'broden':
            pf = SegmentationPrefetcher(data, categories=categories,
                                        once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                        ahead=settings.TALLY_AHEAD, start=start, end=end)
        else:
            pf = CUBSegmentationPrefetcher(data, categories=categories,
                                           once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                           ahead=settings.TALLY_AHEAD, start=start,
                                           end=end)
        # Cache all masks so they can be looked up
        mc = MaskCatalog(pf)

        if savepath and os.path.exists(csvpath):
            print(f"Returning cached {csvpath}")
            return load_csv(csvpath), mc

        g['mc'] = mc
        g['labels'] = mc.labels
        g['masks'] = mc.masks
        g['img2cat'] = mc.img2cat
        g['mask_shape'] = mc.mask_shape

        # Cache label tallies
        g['tally_labels'] = {}
        for lab in tqdm(mc.labels, desc='Tally labels'):
            masks = mc.get_mask(F.Leaf(lab))
            g['tally_labels'][lab] = cmask.area(masks)

        # w/ disjunctions
        tally_units = np.zeros(units, dtype=np.int64)
        g['tally_units'] = tally_units

        # Get unit information (this is expensive (upsampling) and where most of the work is done)
        g['features'] = features
        g['threshold'] = threshold
        mp_args = (
            (u, )
            for u in range(units)
        )
        all_uidx = [None for _ in range(units)]
        all_uhitidx = [None for _ in range(units)]
        pos_labels = [None for _ in range(units)]
        g['pos_labels'] = pos_labels
        g['all_uidx'] = all_uidx
        g['all_uhitidx'] = all_uhitidx
        with mp.Pool(settings.PARALLEL) as p, tqdm(total=units, desc='Tallying units') as pbar:
            for (u, uidx, uhitidx, uhits) in p.imap_unordered(NeuronOperator.get_uhits, mp_args):
                all_uidx[u] = uidx
                all_uhitidx[u] = uhitidx
                # Get all labels which have at least one true here
                label_hits = mc.img2label[uidx].sum(0)
                pos_labels[u] =  np.argwhere(label_hits > 0).squeeze(1)

                tally_units[u] = uhits
                pbar.update()

        # We don't need features anymore
        del g['features']

        records = []

        if settings.UNIT_RANGE is None:
            mp_args = ((u, ) for u in range(units))
            nu = units
        else:
            mp_args = ((u, ) for u in settings.UNIT_RANGE)
            nu = len(settings.UNIT_RANGE)
        with mp.Pool(settings.PARALLEL) as p, tqdm(total=nu, desc='IoU - primitives') as pbar:
            for (u, best, best_noncomp) in p.imap_unordered(NeuronOperator.compute_best_iou, mp_args):
                best_lab, best_iou = best
                best_noncomp_lab, best_noncomp_iou = best_noncomp

                best_name = best_lab.to_str(lambda name: data.name(None, name))
                best_cat = best_lab.to_str(lambda name: categories[pcats[name]])
                best_noncomp_name = best_noncomp_lab.to_str(lambda name: data.name(None, name))
                best_noncomp_cat = best_noncomp_lab.to_str(lambda name: categories[pcats[name]])

                r = {
                    'unit': (u + 1),
                    'category': best_cat,
                    'label': best_name,
                    'score': best_iou,
                    'category_noncomp': best_noncomp_cat,
                    'label_noncomp': best_noncomp_name,
                    'score_noncomp': best_noncomp_iou,
                }
                records.append(r)
                pbar.update()

                if len(records) % 16 == 0:
                    tally_df = pd.DataFrame(records)
                    tally_df.to_csv(csvpath, index=False)

        tally_df = pd.DataFrame(records)
        tally_df.to_csv(csvpath, index=False)
        return records, mc

    @staticmethod
    def compute_best_iou(args):
        u, = args
        best_lab = None
        best_iou = 0.0
        ious = {}
        for lab in g['pos_labels'][u]:
            lab_f = F.Leaf(lab)
            masks = g['masks'][lab]
            lab_iou = NeuronOperator.compute_iou(
                g['all_uidx'][u], g['all_uhitidx'][u], masks, g['tally_units'][u], g['tally_labels'][lab])
            ious[lab] = lab_iou
            if not settings.FORCE_DISJUNCTION and lab_iou > best_iou:
                best_iou = lab_iou
                best_lab = lab_f

        nonzero_iou = Counter({lab: iou for lab, iou in ious.items() if iou > 0})

        if settings.BEAM_SEARCH_LIMIT is not None:
            # Restrict possible candidates
            bs_labs = [t[0] for t in nonzero_iou.most_common(settings.BEAM_SEARCH_LIMIT)]
        else:
            # Search with all possible labels
            bs_labs = g['labels']

        # Beam search
        formulas = {F.Leaf(lab): iou for lab, iou in nonzero_iou.most_common(settings.BEAM_SIZE)}
        best_noncomp = Counter(formulas).most_common(1)[0]

        for i in range(settings.MAX_FORMULA_LENGTH - 1):
            new_formulas = {}
            for formula in formulas:
                for label in bs_labs:
                    for op, negate in [(F.Or, False), (F.And, False), (F.And, True)]:
                        new_term = F.Leaf(label)
                        if negate:
                            new_term = F.Not(new_term)
                        new_term = op(formula, new_term)
                        masks_comp = get_mask_global(g['masks'], new_term)
                        comp_tally_label = cmask.area(masks_comp)
                        comp_iou = NeuronOperator.compute_iou(
                            g['all_uidx'][u], g['all_uhitidx'][u], masks_comp, g['tally_units'][u], comp_tally_label
                        )

                        comp_iou = (settings.FORMULA_COMPLEXITY_PENALTY ** (len(new_term) - 1)) * comp_iou

                        new_formulas[new_term] = comp_iou

            formulas.update(new_formulas)
            # Trim the beam
            formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

        best = Counter(formulas).most_common(1)[0]

        # Get besti ou
        return u, best, best_noncomp

    @staticmethod
    def compute_iou(uidx, uhitidx, masks, tally_unit, tally_label):
        # Compute intersections
        tally_both = cmask.area(cmask.merge((masks, uhitidx), intersect=True))
        iou = (tally_both) / (tally_label + tally_unit - tally_both + 1e-10)
        return iou

    @staticmethod
    def tally_job_std(args):
        features, data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, start, end = args
        units = features.shape[1]
        size_RF = (settings.IMG_SIZE / features.shape[2], settings.IMG_SIZE / features.shape[3])
        fieldmap = ((0, 0), size_RF, size_RF)
        if settings.PROBE_DATASET == 'broden':
            pf = SegmentationPrefetcher(data, categories=data.category_names(),
                                        once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                        ahead=settings.TALLY_AHEAD, start=start, end=end)
        else:
            pf = CUBSegmentationPrefetcher(data, categories=data.category_names(),
                                           once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                           ahead=settings.TALLY_AHEAD, start=start,
                                           end=end)
        count = start
        for batch in tqdm(pf.batches(), desc='Label probing',
                          total=int(np.ceil(end / settings.TALLY_BATCH_SIZE))):

            # Concept map - list of concepts and their associated images
            for concept_map in batch:
                count += 1
                # Get the features of this image
                img_index = concept_map['i']
                scalars, pixels = [], []
                # Loop through the possible categories (color,
                # object, part, scene, texture)
                for cat in data.category_names():
                    # Any labels applying to this image?
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        pixels.append(label_group)
                for scalar in scalars:
                    # Assume it applies to the entire image (this is the area
                    # of the entire label)
                    tally_labels[scalar] += concept_map['sh'] * concept_map['sw']
                if pixels:
                    pixels = np.concatenate(pixels)
                    # Pixels - each number represents the label of the
                    # segmentation mask
                    # Crucially, segmentation masks do NOT overlap
                    tally_label = np.bincount(pixels.ravel())
                    if len(tally_label) > 0:
                        tally_label[0] = 0
                    tally_labels[:len(tally_label)] += tally_label

                # Loop through units (i.e. neurons/channels)
                for unit_id in range(units):
                    # Get the feature of the jth unit across the
                    # entire map (e.g. 7 x 7 map since this is a
                    # high level uniit)
                    feature_map = features[img_index][unit_id]
                    # Check the activation threshold for this unit (e.g. top
                    # 0.5% of activations this unit ever displays, across all
                    # locations of all images in the dataset)
                    # If there is some location where the activation is higher,
                    # compute the binary mask defined by these high activations
                    if feature_map.max() > threshold[unit_id]:

                        # Resize activations w/ interpolation (note - changed
                        # from imresize w/ default bilinear interp; here using
                        # PIL and specify bilinear interp)
                        mask = upsample_features(feature_map, (concept_map['sh'], concept_map['sw']))
                        #reduction = int(round(settings.IMG_SIZE / float(concept_map['sh'])))
                        #mask = upsample.upsampleL(fieldmap, feature_map, shape=(concept_map['sh'], concept_map['sw']), reduction=reduction)
                        indexes = np.argwhere(mask > threshold[unit_id])

                        tally_units[unit_id] += len(indexes)
                        if len(pixels) > 0:
                            # Access the segmasks where our mask is active.
                            # Count how many times each value occurs
                            pixel_ids = pixels[:, indexes[:, 0], indexes[:, 1]].ravel()
                            tally_bt = np.bincount(pixel_ids)
                            if len(tally_bt) > 0:
                                tally_bt[0] = 0
                            # There are 1198 labels. data.labelcat tells you
                            # which category each belongs to (e.g. place, object, etc)
                            # tally_cat groups up the activation and concept
                            # overlap by category and sums them up
                            tally_cat = np.dot(tally_bt[np.newaxis, :], data.labelcat[:len(tally_bt), :])[0]
                            # For this unit, start summing up the TOTAL
                            # activations for each category (again, we're
                            # still not doing labels, but categories!)
                            # Start adding these intersections to "tally both"
                            tally_both[unit_id,:len(tally_bt)] += tally_bt
                        else:
                            # Initialize tally_cat anyways
                            tally_cat = np.zeros(len(data.category_names()),
                                                 dtype=np.float32)
                        for scalar in scalars:
                            tally_cat += data.labelcat[scalar]
                            tally_both[unit_id, scalar] += len(indexes)
                        tally_units_cat[unit_id] += len(indexes) * (tally_cat > 0)


    def tally(self, features, threshold, savepath=''):
        csvpath = os.path.join(settings.OUTPUT_FOLDER, savepath)

        units = features.shape[1]
        labels = len(self.data.label)
        categories = self.data.category_names()
        # hidden units x labels
        tally_both = np.zeros((units, labels), dtype=np.float32)
        # Track the total size of all activation masks (across all images)
        tally_units = np.zeros(units, dtype=np.float32)
        # hidden units x number of CONCEPT categories
        # (i.e. 5)
        tally_units_cat = np.zeros((units, len(categories)), dtype=np.float32)
        tally_labels = np.zeros(labels,dtype=np.float32)

        # Don't parallelize here for now - 1 is fastest for standard; for masks, we'll parallelize in tally_job
        if False and settings.PARALLEL > 1:
            if settings.PROBE_DATASET == 'cub':
                raise NotImplementedError
            psize = int(np.ceil(float(self.data.size()) / settings.PARALLEL))
            ranges = [(s, min(self.data.size(), s + psize)) for s in range(0, self.data.size(), psize) if
                      s < self.data.size()]
            params = [(features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both) + r for r in ranges]
            threadpool = pool.ThreadPool(processes=settings.PARALLEL)
            threadpool.map(NeuronOperator.tally_job, params)
        else:
            maybe_rets = NeuronOperator.tally_job((features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, 0, self.data.size(), savepath, csvpath))

        return maybe_rets
