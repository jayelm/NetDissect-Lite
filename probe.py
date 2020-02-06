import settings
from loader.model_loader import loadmodel
from dissection.neuron import hook_feature, NeuronOperator
from dissection.representation import ReprOperator
from dissection import contrib
from visualize.report import neuron as vneuron, representation as vrepr
from util.clean import clean
from util.misc import safe_layername
from tqdm import tqdm
from scipy.spatial import distance
import torch
import torch.nn.functional as F


def noop(*args, **kwargs):
    return None


hook_modules = []


model = loadmodel(hook_feature, hook_modules=hook_modules)
if settings.LEVEL == 'neuron':
    fo = NeuronOperator()
elif settings.LEVEL == 'representation':
    fo = ReprOperator()
else:
    raise NotImplementedError(settings.LEVEL)

############ STEP 1: feature extraction ###############
# features: list of activations - one 63305 x c x h x w tensor for each feature
# layer (defined by settings.FEATURE_NAMES; default is just layer4)
# maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
features, maxfeature, preds = fo.feature_extraction(model=model)

ranger = tqdm(zip(settings.FEATURE_NAMES, features, maxfeature, preds, [None, *settings.FEATURE_NAMES]),
              total=len(settings.FEATURE_NAMES))

if settings.CONTRIBUTIONS:
    contr, inhib = contrib.get_contributors(hook_modules, alpha_global=0.01)
else:
    contr, inhib = (None, None)

for layer, layer_features, layer_maxfeature, pred, prev_layer in ranger:
    layername = safe_layername(layer)
    prev_layername = safe_layername(prev_layer)
    print(layername, prev_layername)
    ranger.set_description(f'Layer {layername}')
    if settings.LEVEL == 'neuron':
        # ==== STEP 2: Calculate threshold ====
        thresholds = fo.quantile_threshold(layer_features, savepath=f"quantile_{layername}.npy")

        # ==== STEP 3: calculating IoU scores ====
        # Get tally dfname
        if settings.UNIT_RANGE is None:
            tally_dfname = f'tally_{layername}.csv'
        else:
            # Only use a subset of units
            nu = len(settings.UNIT_RANGE)
            tally_dfname = f"tally_{layername}_{min(settings.UNIT_RANGE)}_{max(settings.UNIT_RANGE)}.csv"
        tally_result, mc = fo.tally(layer_features, thresholds, savepath=tally_dfname)

        # ==== STEP 4: generating results ====
        vneuron.generate_html_summary(fo.data, layername, pred, mc,
                                      tally_result=tally_result,
                                      contributors=(contr, inhib),
                                      maxfeature=layer_maxfeature,
                                      features=layer_features,
                                      thresholds=thresholds,
                                      force=True)
    else:
        # Representation (neuralese) search

        # Average responses over image
        layer_features = layer_features.mean((2, 3))

        # ==== STEP 2: Calculate distances and threshold ====
        pdists_condensed = fo.compute_pdists(layer_features)
        thresh = fo.quantile_threshold(pdists_condensed, settings.REPR_ALPHA)
        print(f"Using cutoff threshold of {thresh}")
        sim = pdists_condensed < thresh
        graph = distance.squareform(sim)

        # ==== STEP 3: Search for concepts ====
        records, mc = fo.search_concepts(graph, pred)
        vrepr.generate_html_summary(fo.data, layername, records,
                                    pdists_condensed, pred, mc, thresh, force=True)

    if settings.CLEAN:
        clean()
