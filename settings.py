######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
INDEX_FILE = 'index_ade20k.csv'                # Which index file to use? If _sm, use test mode
CLEAN = False                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'resnet18'                          # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'places365'                       # model trained on: places365, imagenet, or cub. If None,use untrained resnet (random baseline)
PROBE_DATASET = 'broden'                    # which dataset to probe with (broden, cub, or gqa)
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 32                                # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part","scene","texture","color"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
LEVEL = "neuron"    # Neuron or representation
IMAGES = 256 # Representation level - look at this many imagse
REPR_ALPHA = 0.05         # Consider this closest as "close" representations
UNIT_RANGE = None  # Give range if you want to use only some units
MASK_SEARCH = True
FORMULA_COMPLEXITY_PENALTY = 0.99  # How much to downweight formulas by their length
BEAM_SIZE = 5  # Size of the beam when doing formula search
MAX_FORMULA_LENGTH = 3  # Maximum compositional formula length
FORCE_DISJUNCTION = False   # Only output disjunctive concepts. (Otherwise, disjunctive concepts are only identified if they have the highest IoU relative to other categories)

INDEX_SUFFIX = INDEX_FILE.split('index')[1].split('.csv')
if not INDEX_SUFFIX:
    INDEX_SUFFIX = ''
else:
    INDEX_SUFFIX = INDEX_SUFFIX[0]

OUTPUT_FOLDER = f"result/{MODEL}_{DATASET}_{PROBE_DATASET}{INDEX_SUFFIX}_{LEVEL}_{MAX_FORMULA_LENGTH}"  # result will be stored in this folder

print(OUTPUT_FOLDER)

TEST_MODE = INDEX_FILE == 'index_sm.csv'

assert LEVEL in {'neuron', 'representation'}

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision; <UNTRAINED> uses untrained torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if PROBE_DATASET == 'broden':
    if MODEL != 'alexnet':
        DATA_DIRECTORY = 'dataset/broden1_224'
        IMG_SIZE = 224
    else:
        DATA_DIRECTORY = 'dataset/broden1_227'
        IMG_SIZE = 227
elif PROBE_DATASET == 'cub':
    if MODEL != 'alexnet':
        DATA_DIRECTORY = 'dataset/CUB_200_2011'
        IMG_SIZE = 224
    else:
        raise NotImplementedError
elif PROBE_DATASET == 'gqa':
    DATA_DIRECTORY = 'dataset/gqa'
    IMG_SIZE = 224
else:
    raise NotImplementedError(f"Unknown dataset {PROBE_DATASET}")

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
    elif DATASET == 'cub':
        MODEL_FILE = 'zoo/trained/resnet18_cub_finetune/model_best.pth'
        MODEL_PARALLEL = False
    elif DATASET is None:
        MODEL_FILE = '<UNTRAINED>'
        MODEEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
else:
    WORKERS = 12
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
