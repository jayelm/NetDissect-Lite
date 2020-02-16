import settings
from loader.model_loader import loadmodel
from dissection.neuron import hook_feature, NeuronOperator
from dissection.representation import ReprOperator
from dissection import contrib
from visualize.report import neuron as vneuron, representation as vrepr, final as vfinal
from util.clean import clean
from util.misc import safe_layername
from tqdm import tqdm
from scipy.spatial import distance
import torch
import torch.nn.functional as F
import pickle
import os
import pandas as pd
from loader.data_loader import ade20k


def noop(*args, **kwargs):
    return None


layernames = list(map(safe_layername, settings.FEATURE_NAMES))


hook_modules = []


def spread_contrs(weights, contrs, layernames):
    """
    [
    (layer1)
    {
        'feat_corr': {
            'weight': ...,
            'contr': ...
        }
    }
    ]
    """
    return [
        {name: {
            'weight': weights[name][i],
            'contr': contrs[name][i]
        } for name in weights.keys()}
        for i in range(len(layernames))
    ]

model = loadmodel(hook_feature, hook_modules=hook_modules)
if settings.LEVEL == 'neuron':
    fo = NeuronOperator()
elif settings.LEVEL == 'representation':
    fo = ReprOperator()
else:
    raise NotImplementedError(settings.LEVEL)

# ==== STEP 1: Feature extraction ====
# features: list of activations - one 63305 x c x h x w tensor for each feature
# layer (defined by settings.FEATURE_NAMES; default is just layer4)
# maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
features, maxfeature, preds, logits = fo.feature_extraction(model=model)

# ==== STEP 1.5: confusion matrix =====
pr = preds[-1]
pred_records = []
for i, (p, t) in enumerate(pr):
    pred_name = ade20k.I2S[p]
    target_name = f'{fo.data.scene(i)}-s'
    if target_name in ade20k.S2I:
        pred_records.append((pred_name, target_name))

pred_df = pd.DataFrame.from_records(pred_records, columns=['pred', 'target'])
pred_df.to_csv(os.path.join(settings.OUTPUT_FOLDER, 'preds.csv'), index=False)
print(f"Accuracy: {(pred_df.pred == pred_df.target).mean() * 100:.2f}%")

# ==== STEP 2: Threshold quantization ====
thresholds = [fo.quantile_threshold(lf, savepath=f'quantile_{ln}.npy')
              for lf, ln in zip(features, layernames)]

# ==== New: multilayer case - neuron contributions ====
if settings.CONTRIBUTIONS:
    contr_fname = os.path.join(settings.OUTPUT_FOLDER, 'contrib.pkl')
    if os.path.exists(contr_fname):
        print(f"Loading cached contributions {contr_fname}")
        with open(contr_fname, 'rb') as f:
            contrs_spread = pickle.load(f)
    else:
        print("Computing contributions")
        # TODO: Maybe multiprocess this if it ends up being really slow?
        weights = {
            'weight': contrib.get_weights(hook_modules),
            'feat_corr': contrib.get_feat_corr(features),
            'act_iou': contrib.get_act_iou(features, thresholds)
        }
        contrs = {
            name: contrib.threshold_contributors(weight, alpha_global=0.01)
            for name, weight in weights.items()
        }
        contrs_spread = spread_contrs(weights, contrs, layernames)
        with open(contr_fname, 'wb') as f:
            pickle.dump(contrs_spread, f)
else:
    contrs_spread = [
        {} for _ in settings.FEATURE_NAMES
    ]

# Zip it all together
ranger = tqdm(zip(layernames, features, maxfeature, thresholds, preds, [None, *layernames],
                  contrs_spread),
              total=len(layernames))

prev_tally = None
for layername, layer_features, layer_maxfeature, layer_thresholds, layer_preds, prev_layername, layer_contrs in ranger:
    ranger.set_description(f'Layer {layername}')
    if settings.LEVEL == 'neuron':

        # ==== STEP 3: calculating IoU scores ====
        # Get tally dfname
        if settings.UNIT_RANGE is None:
            tally_dfname = f'tally_{layername}.csv'
        else:
            # Only use a subset of units
            tally_dfname = f"tally_{layername}_{min(settings.UNIT_RANGE)}_{max(settings.UNIT_RANGE)}.csv"
        tally_result, mc = fo.tally(layer_features, layer_thresholds, savepath=tally_dfname)

        # ==== STEP 4: generating results ====
        vneuron.generate_html_summary(fo.data, layername, layer_preds, mc,
                                      tally_result=tally_result,
                                      contributors=layer_contrs,
                                      maxfeature=layer_maxfeature,
                                      features=layer_features,
                                      prev_layername=prev_layername,
                                      prev_tally=prev_tally,
                                      thresholds=layer_thresholds,
                                      force=False)

        prev_tally = {
            record['unit']: record['label'] for record in tally_result
        }
    else:
        # Representation (neuralese) search

        # Average responses over image
        layer_features = layer_features.mean((2, 3))

        # ==== STEP 2: Calculate distances and threshold ====
        pdists_condensed = fo.compute_pdists(layer_features, fname=f'pdists_{layername}.npz')
        thresh = fo.quantile_threshold(pdists_condensed, settings.REPR_ALPHA)
        print(f"Using cutoff threshold of {thresh}")
        sim = pdists_condensed < thresh
        graph = distance.squareform(sim)

        # ==== STEP 3: Search for concepts ====
        if settings.IMAGES is None:
            input_fname = f'input_{layername}.csv'
        else:
            input_fname = f'input_{layername}_{settings.IMAGES}.csv'
        records, mc = fo.search_concepts(graph, layer_preds, fname=input_fname)
        vrepr.generate_html_summary(fo.data, layername, records,
                                    pdists_condensed, layer_preds, mc, thresh, force=True)

# ==== STEP 5: generate last layer ====
if settings.LEVEL == 'neuron':
    # Final layer - neurons that contribute to decisions
    final_weight_np = model.fc.weight.detach().cpu().numpy()
    weights = {
        'weight': [None, final_weight_np],
        'feat_corr': contrib.get_feat_corr([features[-1].mean(2).mean(2), logits[-1]],
                                           flattened=True),
        # TODO: For these, impl features for weight (quantile)
        #  'act_iou': contrib.get_act_iou(features, thresholds),
        #  'act_iou_inhib': contrib.get_act_iou_inhib(features, thresholds)
    }
    contrs = {
        name: contrib.threshold_contributors(weight, alpha_global=0.01)
        for name, weight in weights.items()
    }
    contrs_spread = spread_contrs(weights, contrs, layernames)
    vfinal.generate_final_layer_summary(fo.data, final_weight_np, features[-1], thresholds[-1], preds[-1], logits[-1], prev_layername=layernames[-1], prev_tally=prev_tally, contributors=contrs_spread)

if settings.CLEAN:
    clean()
