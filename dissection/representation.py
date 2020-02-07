import os
from PIL import Image
import numpy as np
import torch
import settings
import util.upsample as upsample
import pandas as pd
import util.vecquantile as vecquantile
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
from dissection.neuron import NeuronOperator


from pycocotools import mask as cmask
from scipy.spatial.distance import squareform, jaccard

# Janky - globals for multiprocessing to prevent shared memory
g = {}

features_blobs = []


def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    res = n*j - j*(j+1)/2 + i - 1 - j
    if int(res) != res:
        raise ValueError(f"Got non-integere value {res} for ({i}, {j}, {n})")
    return int(res)


class ReprOperator(NeuronOperator):
    def compute_pdists(self, features):
        pdists_fname = os.path.join(settings.OUTPUT_FOLDER, 'pdists.npz')
        if os.path.exists(pdists_fname):
            print(f"Loading cached {pdists_fname}")
            return np.load(pdists_fname)['arr_0']
        else:
            print(f"Computing pdists")
            lf_t = torch.from_numpy(features)
            if settings.GPU:
                lf_t = lf_t.cuda()
            with torch.no_grad():
                pdists = torch.cdist(lf_t, lf_t)
            pdists_np = pdists.cpu().numpy()
            # Condense
            pdists_np = squareform(pdists_np)
            np.savez_compressed(pdists_fname, pdists_np)
            return pdists_np

    def quantile_threshold(self, pdists, percentile):
        if percentile > 1:
            raise ValueError("Specific percentile between 0 and 1")
        return np.percentile(pdists, percentile * 100)

    def compute_adj_list(self, graph):
        adj = {}
        for i in range(graph.shape[0]):
            adj[i] = np.argwhere(graph[i]).squeeze(1)
        return adj

    def search_concepts(self, graph, preds):
        if settings.IMAGES is None:
            max_i = graph.shape[0]
            sfx = ''
        else:
            max_i = settings.IMAGES
            sfx = f'_{max_i}'
        input_fname = os.path.join(settings.OUTPUT_FOLDER, f'input{sfx}.csv')

        categories = self.data.category_names()
        pcats = self.data.primary_categories_per_index()

        if settings.PROBE_DATASET == 'broden':
            pf = SegmentationPrefetcher(self.data, categories=categories,
                                        once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                        ahead=settings.TALLY_AHEAD, start=0, end=self.data.size())
        else:
            pf = CUBSegmentationPrefetcher(self.data, categories=categories,
                                           once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                           ahead=settings.TALLY_AHEAD, start=0,
                                           end=self.data.size())

        mc = MaskCatalog(pf, cache=False, rle=False)

        if os.path.exists(input_fname):
            print(f"Returning cached {input_fname}")
            input_df = pd.read_csv(input_fname)
            return input_df.to_dict('records'), mc

        g['graph'] = graph
        g['classes'] = mc.classes
        g['n_classes'] = mc.n_classes
        g['label2img'] = mc.img2label.T
        g['n_labels'] = g['label2img'].shape[0]

        records = []

        random = np.random.RandomState(seed=settings.SEED)
        i_rand = random.permutation(graph.shape[0])
        mp_args = [(i_rand[i], ) for i in range(max_i)]
        with mp.Pool(settings.PARALLEL) as p, tqdm(total=max_i, desc='Images') as pbar:
            for i, best, best_noncomp in p.imap_unordered(ReprOperator.compute_best_label, mp_args):
                # Name the label
                best['label'], best['category'] = (
                    best['label'].to_str(lambda name: self.data.name(None, name)),
                    best['label'].to_str(lambda name: categories[pcats[name]])
                )

                best_noncomp['label'], best_noncomp['category'] = (
                    best_noncomp['label'].to_str(lambda name: self.data.name(None, name)),
                    best_noncomp['label'].to_str(lambda name: categories[pcats[name]])
                )
                best_noncomp = {f'{k}_noncomp': v for k, v in best_noncomp.items()}

                r = {
                    'input': i,
                    'pred_label': preds[i, 0],
                    'true_label': preds[i, 1],
                    'correct': preds[i, 0] == preds[i, 1],
                    **best,
                    **best_noncomp,
                }
                records.append(r)
                pbar.update()

                if len(records) % 32 == 0:
                    # Save every 32
                    res_df = pd.DataFrame(records)
                    res_df.to_csv(input_fname,
                                  index=False)

        res_df = pd.DataFrame(records)
        res_df.to_csv(input_fname,
                      index=False)
        return records, mc

    @staticmethod
    def compute_best_label(args):
        i, = args
        links = g['graph'][i]

        isims = {}
        for lab in range(g['n_labels']):
            isims[lab] = 1 - jaccard(links, g['label2img'][lab])

        formulas = {F.Leaf(lab): iou for lab, iou in Counter(isims).most_common(settings.BEAM_SIZE)}

        best_noncomp = Counter(formulas).most_common(1)[0]

        for i in range(settings.MAX_FORMULA_LENGTH - 1):
            # TODO: Beam search
            new_formulas = {}
            for formula in formulas:
                for lab in range(g['n_labels']):
                    for op, negate in [(F.Or, False), (F.And, False), (F.And, True)]:
                        new_term = F.Leaf(lab)
                        if negate:
                            new_term = F.Not(new_term)
                        new_term = op(formula, new_term)
                        labels_comp = ReprOperator.get_labels(new_term)
                        comp_sim = 1 - jaccard(links, labels_comp)
                        comp_sim *= (settings.FORMULA_COMPLEXITY_PENALTY ** (len(new_term) - 1))

                        new_formulas[new_term] = comp_sim
            formulas.update(new_formulas)
            # Trim the beam
            formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

        best = Counter(formulas).most_common(1)[0]

        best = {
            'label': best[0],
            'score': best[1],
            **ReprOperator.compute_label_statistics(links, best[0])
        }

        best_noncomp = {
            'label': best_noncomp[0],
            'score': best_noncomp[1],
            **ReprOperator.compute_label_statistics(links, best_noncomp[0])
        }

        return args[0], best, best_noncomp


    @staticmethod
    def compute_label_statistics(links, lab):
        """
        Compute some label statistics
        """
        neighborhood_coverage = links.mean()
        neighborhood_class_coverage = links.mean()

        labels = ReprOperator.get_labels(lab)
        label_coverage = labels.mean()
        label_class_coverage = len(np.unique(g['classes'][labels])) / g['n_classes']

        return {
            'neighborhood_coverage': neighborhood_coverage,
            'neighborhood_class_coverage': neighborhood_class_coverage,
            'label_coverage': label_coverage,
            'label_class_coverage': label_class_coverage,
        }


    @staticmethod
    def get_labels(f, labels=None):
        """
        Serializable/global version of get_mask for multiprocessing
        """
        # TODO: Handle here when doing AND and ORs of scenes vs scalars.
        if isinstance(f, F.And):
            labels_l = ReprOperator.get_labels(f.left, labels=labels)
            labels_r = ReprOperator.get_labels(f.right, labels=labels)
            return np.logical_and(labels_l, labels_r)
        elif isinstance(f, F.Or):
            labels_l = ReprOperator.get_labels(f.left, labels=labels)
            labels_r = ReprOperator.get_labels(f.right, labels=labels)
            return np.logical_or(labels_l, labels_r)
        elif isinstance(f, F.Not):
            labels_val = ReprOperator.get_labels(f.val, labels=labels)
            return np.logical_not(labels_val)
        elif isinstance(f, F.Leaf):
            if labels is None:
                return g['label2img'][f.val]
            else:
                return labels[f.val]
        else:
            raise ValueError("Most be passed formula")
