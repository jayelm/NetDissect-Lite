import os
from PIL import Image
import numpy as np
import torch
import settings
import util.upsample as upsample
import util.vecquantile as vecquantile
import multiprocessing.pool as pool
from loader.data_loader.broden import load_csv
from loader.data_loader.broden import SegmentationData, SegmentationPrefetcher
from tqdm import tqdm, trange
import csv

features_blobs = []
def hook_feature(module, inp, output):
    features_blobs.append(output.data.cpu().numpy())


class FeatureOperator:

    def __init__(self):
        if not os.path.exists(settings.OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'image'))
        self.data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATAGORIES)
        self.loader = SegmentationPrefetcher(self.data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
        self.mean = [109.5388,118.6897,124.6901]

    def feature_extraction(self, model=None, memmap=True):
        loader = self.loader
        # extract the max value activaiton for each image
        maxfeatures = [None] * len(settings.FEATURE_NAMES)
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files =  [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name)  for feature_name in  settings.FEATURE_NAMES]
            mmap_max_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_max.mmap" % feature_name) for feature_name in settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file) in enumerate(zip(mmap_files,mmap_max_files)):
                if os.path.exists(mmap_file) and os.path.exists(mmap_max_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=np.float32, mode='r', shape=tuple(features_size[i]))
                    maxfeatures[i] = np.memmap(mmap_max_file, dtype=np.float32, mode='r', shape=tuple(features_size[i][:2]))
                else:
                    print('file missing, loading from scratch')
                    skip = False
            if skip:
                return wholefeatures, maxfeatures

        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        for batch_idx,batch in tqdm(enumerate(loader.tensor_batches(bgr_mean=self.mean)), desc='Extracting features', total=int(np.ceil(num_batches))):
            del features_blobs[:]
            inp = batch[0]
            batch_size = len(inp)
            inp = torch.from_numpy(inp[:, ::-1, :, :].copy())
            inp.div_(255.0 * 0.224)
            if settings.GPU:
                inp = inp.cuda()
            logit = model.forward(inp)
            while np.isnan(logit.data.cpu().max()):
                print("nan") #which I have no idea why it will happen
                del features_blobs[:]
                logit = model.forward(inp)
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
        if len(feat_batch.shape) == 2:
            wholefeatures = maxfeatures
        return wholefeatures,maxfeatures

    def quantile_threshold(self, features, savepath=''):
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
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
        features, data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, start, end = args
        units = features.shape[1]
        size_RF = (settings.IMG_SIZE / features.shape[2], settings.IMG_SIZE / features.shape[3])
        fieldmap = ((0, 0), size_RF, size_RF)
        pd = SegmentationPrefetcher(data, categories=data.category_names(),
                                    once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                    ahead=settings.TALLY_AHEAD, start=start, end=end)
        count = start
        for batch in tqdm(pd.batches(), desc='Label probing',
                          total=int(np.ceil(end / settings.TALLY_BATCH_SIZE))):

            # Concept map - list of concepts and their associated
            # images
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
                    tally_labels[scalar] += concept_map['sh'] * concept_map['sw']
                if pixels:
                    pixels = np.concatenate(pixels)
                    # Pixels - each number represents the label of the
                    # segmentation mask
                    # XXX: Crucially, segmentation masks do NOT overlap
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
                        mask = np.array(Image.fromarray(feature_map).resize((concept_map['sh'], concept_map['sw']), resample=Image.BILINEAR))
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
                        for scalar in scalars:
                            tally_cat += data.labelcat[scalar]
                            tally_both[unit_id, scalar] += len(indexes)
                        tally_units_cat[unit_id] += len(indexes) * (tally_cat > 0)


    def tally(self, features, threshold, savepath=''):
        csvpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(csvpath):
            return load_csv(csvpath)

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

        if settings.PARALLEL > 1:
            psize = int(np.ceil(float(self.data.size()) / settings.PARALLEL))
            ranges = [(s, min(self.data.size(), s + psize)) for s in range(0, self.data.size(), psize) if
                      s < self.data.size()]
            params = [(features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both) + r for r in ranges]
            threadpool = pool.ThreadPool(processes=settings.PARALLEL)
            threadpool.map(FeatureOperator.tally_job, params)
        else:
            FeatureOperator.tally_job((features, self.data, threshold, tally_labels, tally_units, tally_units_cat, tally_both, 0, self.data.size()))

        primary_categories = self.data.primary_categories_per_index()
        # Tally units cat - how often is the neuron activated FOR EACH
        # CATEGORY, not label, across all images.
        # Dotting with labelcat gives us (n_units x n_labels) tensor -
        # for each unit, for each label, this is the total number of times the
        # neuron was activated for ALL labels OF A SPECIFIC CATEGORY (Across
        # images).
        # Because the data set
        # contains some types of labels which are not present on some
        # subsets of inputs, the sums are computed only on the subset
        # of images that have at least one labeled concept of the same
        # category as c
        tally_units_cat = np.dot(tally_units_cat, self.data.labelcat.T)

        # ==== DISJUNCTIONS ====
        disj_iou = []
        disj_lab = []
        disj_name = []
        disj_tally_both = []
        disj_tally_units_cat = []
        disj_tally_labels = []
        for unit_id in trange(units, desc='Disjunctions'):
            tally_bt_uid = tally_both[unit_id]
            labs_uid = np.argwhere(tally_bt_uid != 0.0).squeeze(1)
            cats_uid =  primary_categories[labs_uid]

            tally_bt_uid = np.add.outer(tally_bt_uid[labs_uid], tally_bt_uid[labs_uid])

            tally_uc_uid = tally_units_cat[unit_id]
            tally_uc_uid = np.add.outer(tally_uc_uid[labs_uid], tally_uc_uid[labs_uid])

            tally_labs_uid = np.add.outer(tally_labels[labs_uid], tally_labels[labs_uid])
            iou_uid = tally_bt_uid / (tally_uc_uid + tally_labs_uid - tally_bt_uid + 1e-10)
            # We don't care about "X v X" disjuncts
            np.fill_diagonal(iou_uid, 0.0)
            names_uid = [self.data.name(None, c) for c in labs_uid]
            iou_labs_flat = []
            iou_names_flat = []
            iou_flat = []
            tally_bt_flat = []
            tally_uc_flat = []
            tally_labs_flat = []
            for i, (lab1, name1) in enumerate(zip(labs_uid, names_uid)):
                for j, (lab2, name2) in enumerate(zip(labs_uid[:i], names_uid[:i])):  # lower tri
                    iou_labs_flat.append([lab1, lab2])
                    name_disj = f'{name1} OR {name2}'
                    iou_names_flat.append(name_disj)
                    iou_flat.append(iou_uid[i, j])
                    tally_bt_flat.append(tally_bt_uid[i, j])
                    tally_uc_flat.append(tally_uc_uid[i, j])
                    tally_labs_flat.append(tally_labs_uid[i, j])
            iou_labs_flat = np.array(iou_labs_flat)
            iou_flat = np.array(iou_flat)
            iou_names_flat = np.array(iou_names_flat)
            tally_bt_flat = np.array(tally_bt_flat)
            tally_uc_flat = np.array(tally_uc_flat)
            tally_labs_flat = np.array(tally_labs_flat)

            # What do I do about categories at this point?
            # TODO: Ignoring categories right now. Just pick the best
            # disjunction across all categories.
            disj_idx = iou_flat.argmax()
            disj_iou.append(iou_flat[disj_idx])
            disj_lab.append(iou_labs_flat[disj_idx])
            disj_name.append(iou_names_flat[disj_idx])
            disj_tally_both.append(tally_bt_flat[disj_idx])
            disj_tally_units_cat.append(tally_uc_flat[disj_idx])
            disj_tally_labels.append(tally_labs_flat[disj_idx])

        disj_iou = np.array(disj_iou, dtype=np.float32)
        disj_lab = np.array(disj_lab, dtype=np.int)
        disj_name = np.array(disj_name)
        disj_tally_both = np.array(disj_tally_both)
        disj_tally_units_cat = np.array(disj_tally_units_cat)
        disj_tally_labels = np.array(disj_tally_labels)

        # NUMERATOR: tally both - magnitude of intersection of activation masks
        # and label masks
        # DENOMINATOR: union - sum of total activations for each neuron (within
        # the broad category) plus sum of total pixels covered by the label
        # masks minus the intersection overlap
        # Therefore the ij-th value is iou_i,j , the ratio. This should always
        # be \in [0, 1]
        # FIXME: Should I actually be using ints due to size?
        iou = tally_both / (tally_units_cat + tally_labels[np.newaxis, :] - tally_both + 1e-10)
        # We should get the best labels now - for each neuron pick the label with the highest iou.
        # For each category type [0, 1, 2, 3, 4]:
        # (n_categories x n_units x n_labels) matrix
        # Where each row are the iou values but only for labels in the category
        # Basically splitting up iou into these partially masked matrices
        # pciou -> Per-Category IOU
        pciou = np.array([iou * (primary_categories[np.arange(iou.shape[1])] == ci)[np.newaxis, :] for ci in range(len(self.data.category_names()))])
        # For each category/each unit, which label was the most active?
        label_pciou = pciou.argmax(axis=2)
        # Get the NAME of the label as in label_pciou.
        name_pciou = [
            [self.data.name(None, j) for j in label_pciou[ci]]
            for ci in range(len(label_pciou))]
        # The SCORE for each best label_pciou
        # Pretty sure this can just be replaced with pciou.max(2)
        score_pciou = pciou[
            np.arange(pciou.shape[0])[:, np.newaxis],
            np.arange(pciou.shape[1])[np.newaxis, :],
            label_pciou]
        # Add the 6th disjunction category
        score_pciou = np.concatenate([score_pciou, disj_iou[np.newaxis]], 0)
        if settings.FORCE_DISJUNCTION:
            bestcat_pciou = np.full((1, units),
                                    score_pciou.shape[0] - 1, dtype=np.int64)
            ordering = score_pciou[-1].argsort()[::-1]
        else:
            # Finally, we pick only the best category
            bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
            # Arrange units by their overall best cat iou score
            ordering = score_pciou.max(axis=0).argsort()[::-1]
        rets = []

        for i,unit in enumerate(ordering):
            # Top images are top[unit]
            bestcat = bestcat_pciou[0, unit]
            if bestcat == score_pciou.shape[0] - 1:
                # Disjunction
                data = {
                    'unit': (unit + 1),
                    'category': 'disj',
                    'label': disj_name[unit],
                    'score': disj_iou[unit]
                }
            else:
                data = {
                    'unit': (unit + 1),
                    'category': categories[bestcat],
                    'label': name_pciou[bestcat][unit],
                    'score': score_pciou[bestcat][unit]
                }

            for ci, cat in enumerate(categories + ['disj']):
                if cat == 'disj':
                    data.update({
                        f'{cat}-label': disj_name[unit],
                        f'{cat}-truth': disj_tally_labels[unit],
                        f'{cat}-activation': disj_tally_units_cat[unit],
                        f'{cat}-intersect': disj_tally_both[unit],
                        f'{cat}-iou': disj_iou[unit]
                    })
                else:
                    label = label_pciou[ci][unit]
                    data.update({
                        f'{cat}-label': name_pciou[ci][unit],
                        f'{cat}-truth': tally_labels[label],
                        f'{cat}-activation': tally_units_cat[unit, label],
                        f'{cat}-intersect': tally_both[unit, label],
                        f'{cat}-iou': score_pciou[ci][unit]
                    })
            rets.append(data)

        if savepath:
            csv_fields = sum([[
                f'{cat}-label',
                f'{cat}-truth',
                f'{cat}-activation',
                f'{cat}-intersect',
                f'{cat}-iou'] for cat in categories + ['disj']],
                ['unit', 'category', 'label', 'score'])
            with open(csvpath, 'w') as f:
                writer = csv.DictWriter(f, csv_fields)
                writer.writeheader()
                for i in range(len(ordering)):
                    writer.writerow(rets[i])
        return rets
