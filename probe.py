import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from repr_operation import ReprOperator
from visualize.report import neuron as vneuron, representation as vrepr
from util.clean import clean
from tqdm import tqdm
from scipy.spatial import distance
import torch
import torch.nn.functional as F


def noop(*args, **kwargs):
    return None


model = loadmodel(hook_feature)
if settings.LEVEL == 'neuron':
    fo = FeatureOperator()
elif settings.LEVEL == 'representation':
    fo = ReprOperator()
else:
    raise NotImplementedError(settings.LEVEL)

############ STEP 1: feature extraction ###############
# features: list of activations - one 63305 x c x h x w tensor for each feature
# layer (defined by settings.FEATURE_NAMES; default is just layer4)
# maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
features, maxfeature = fo.feature_extraction(model=model)

ranger = tqdm(zip(settings.FEATURE_NAMES, features, maxfeature),
              total=len(settings.FEATURE_NAMES))
for layer, layer_features, layer_maxfeature in ranger:
    ranger.set_description(f'Layer {layer}')
    if settings.LEVEL == 'neuron':
        #### STEP 2: Calculate threshold
        thresholds = fo.quantile_threshold(layer_features, savepath="quantile.npy")

        #### STEP 3: calculating IoU scores
        tally_result, mc = fo.tally(layer_features, thresholds, savepath="tally.csv")

        #### STEP 4: generating results
        vneuron.generate_html_summary(fo.data, layer, mc,
                                      tally_result=tally_result,
                                      maxfeature=layer_maxfeature,
                                      features=layer_features,
                                      thresholds=thresholds,
                                      force=True)
    else:
        # Go through each input. Find inputs nearest to each other according to some threshold (can do this unsupervised by doing w/in sum squares)
        # Average responses over image
        layer_features = layer_features.mean((2, 3))
        #### STEP 2: Calculate distances and threshold
        pdists_condensed = fo.compute_pdists(layer_features)
        thresh = fo.quantile_threshold(pdists_condensed, settings.REPR_ALPHA)
        print(f"Using cutoff threshold of {thresh}")
        sim = pdists_condensed < thresh
        graph = distance.squareform(sim)
        #  adj = fo.compute_adj_list(graph)
        #### STEP 3: Search for concepts
        records, mc = fo.search_concepts(graph)
        vrepr.generate_html_summary(fo.data, layer, records,
                                    pdists_condensed, mc, thresh, force=True)

    if settings.CLEAN:
        clean()
