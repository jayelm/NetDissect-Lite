import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
from tqdm import tqdm

fo = FeatureOperator()
model = loadmodel(hook_feature)

############ STEP 1: feature extraction ###############
# features: list of activations - one 63305 x c x h x w tensor for each feature
# layer (defined by settings.FEATURE_NAMES; default is just layer4)
# maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
features, maxfeature = fo.feature_extraction(model=model)

ranger = tqdm(zip(settings.FEATURE_NAMES, features, maxfeature),
              total=len(settings.FEATURE_NAMES))
for layer, layer_features, layer_maxfeature in ranger:
    ranger.set_description(f'Layer {layer}')
    #### STEP 2: Calculate threshold
    thresholds = fo.quantile_threshold(layer_features, savepath="quantile.npy")

    #### STEP 3: calculating IoU scores
    tally_result = fo.tally(layer_features, thresholds, savepath="tally.csv")

    #### STEP 4: generating results
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=layer_maxfeature,
                          features=layer_features,
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()
