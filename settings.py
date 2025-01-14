######### global settings  #########
GPU = True  # running on GPU is highly suggested
SEED = 42  # Used mainly to seed random images for now.
INDEX_FILE = "index_ade20k.csv"  # Which index file to use? If _sm, use test mode
CLEAN = False  # set to "True" if you want to clean the temporary large files after generating result
MODEL = "resnet18"  # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = "ade20k"  # model trained on: places365, imagenet, or cub. If None,use untrained resnet (random baseline)
MODEL_CHECKPOINT = 0  # used only for ade20k - model training checkpoint
CONV4_NUM_CHANNELS = 256
PROBE_DATASET = "broden"  # which dataset to probe with (broden, cub, or gqa)
QUANTILE = 0.005  # the threshold used for activation
SEG_THRESHOLD = 0.04  # the threshold used for visualization
SCORE_THRESHOLD = 0.04  # the threshold used for IoU score (in HTML file)
CONTRIBUTIONS = True  # If True, assume successive layers feed into each other; will use weights of layer i+1 to identify neurons contributing to layer i
TOPN = 5  # to show top N image with highest activation for each unit
PARALLEL = (
    8  # how many process is used for tallying (Experiments show that 1 is the fastest)
)
CATEGORIES = [
    "object",
    "part",
    "scene",
    "texture",
    "color",
]  # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
LEVEL = "neuron"  # Neuron or representation
IMAGES = 256  # Representation level - look at this many imagse
REPR_ALPHA = 0.05  # Consider this closest as "close" representations
UNIT_RANGE = None  # Give range if you want to use only some units
EMBEDDING_SUMMARY = (
    True  # Make an embedding-based summary of the formulas during visualization
)
WN_SUMMARY = (
    False  # Make an embedding-based summary of the formulas during visualization
)
SEMANTIC_CONSISTENCY = True  # Evaluate semantic consistency of formulas
FORMULA_COMPLEXITY_PENALTY = 1.00  # How much to downweight formulas by their length
BEAM_SEARCH_LIMIT = 50  # (artificially) limit beam to this many candidates
BEAM_SIZE = 5  # Size of the beam when doing formula search
MAX_FORMULA_LENGTH = 10  # Maximum compositional formula length
TREE_MAXDEPTH = 4  # Index tree depth
TREE_MAXCHILDREN = 3  # Index tree max children
TREE_UNITS = range(1, 365, 10)  # How many units to build tree for
FORCE_DISJUNCTION = False  # Only output disjunctive concepts. (Otherwise, disjunctive concepts are only identified if they have the highest IoU relative to other categories)

INDEX_SUFFIX = INDEX_FILE.split("index")[1].split(".csv")
if PROBE_DATASET != "broden" or not INDEX_SUFFIX:
    INDEX_SUFFIX = ""
else:
    INDEX_SUFFIX = INDEX_SUFFIX[0]

TEST_MODE = INDEX_FILE == "index_sm.csv"

if MODEL == 'conv4':
    mbase = f"{MODEL}_{CONV4_NUM_CHANNELS}"
else:
    mbase = MODEL
OUTPUT_FOLDER = f"result/{mbase}_{DATASET}_{PROBE_DATASET}{INDEX_SUFFIX}_{LEVEL}_{MAX_FORMULA_LENGTH}{'_test' if TEST_MODE else ''}{f'_checkpoint_{MODEL_CHECKPOINT}' if DATASET == 'ade20k' else ''}"  # result will be stored in this folder

print(OUTPUT_FOLDER)

assert LEVEL in {"neuron", "representation"}

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

if PROBE_DATASET == "broden":
    if MODEL != "alexnet":
        DATA_DIRECTORY = "dataset/broden1_224"
        IMG_SIZE = 224
    else:
        DATA_DIRECTORY = "dataset/broden1_227"
        IMG_SIZE = 227
elif PROBE_DATASET == "cub":
    if MODEL != "alexnet":
        DATA_DIRECTORY = "dataset/CUB_200_2011"
        IMG_SIZE = 224
    else:
        raise NotImplementedError
elif PROBE_DATASET == "gqa":
    DATA_DIRECTORY = "dataset/gqa"
    IMG_SIZE = 224
else:
    raise NotImplementedError(f"Unknown dataset {PROBE_DATASET}")

if DATASET == "places365":
    NUM_CLASSES = 365
elif DATASET == "imagenet":
    NUM_CLASSES = 1000
elif DATASET == "cub":
    NUM_CLASSES = 200
elif DATASET == "ade20k":
    NUM_CLASSES = 365

if MODEL not in {"resnet18", "renset101", "conv4", "alexnet", "vgg16"}:
    raise NotImplementedError(f"model = {MODEL}")

if MODEL == "resnet18":
    FEATURE_NAMES = [
        #  ['layer2', '0', 'conv1'], ['layer2', '0', 'conv2'],
        #  ['layer2', '1', 'conv1'], ['layer2', '1', 'conv2'],
        #  ['layer3', '0', 'conv1'], ['layer3', '0', 'conv2'],
        #  ['layer3', '1', 'conv1'], ['layer3', '1', 'conv2'],
        #  ['layer4', '0', 'conv1'], ['layer4', '0', 'conv2'],
        #  ["layer4", "1", "conv1"],
        #  ["layer4", "1", "conv2"],
        'layer4'
    ]
    #  FEATURE_NAMES = ['layer4']
elif MODEL == "resnet101":
    FEATURE_NAMES = ["layer4"]
elif MODEL == "conv4":
    # Not sure...
    FEATURE_NAMES = [["trunk", "3"]]
elif MODEL == "alexnet":
    # Not sure...
    FEATURE_NAMES = ["layer4"]
elif MODEL == "vgg16":
    # Not sure...
    FEATURE_NAMES = ["layer4"]

if DATASET == "places365":
    MODEL_FILE = f"zoo/{MODEL}_places365.pth.tar"
    MODEL_PARALLEL = True
elif DATASET == "imagenet":
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif DATASET == "cub":
    MODEL_FILE = f"zoo/trained/{MODEL}_cub_finetune/model_best.pth"
    MODEL_PARALLEL = False
elif DATASET == "ade20k":
    MODEL_FILE = f"zoo/trained/{mbase}_ade20k_finetune/{MODEL_CHECKPOINT}.pth"
    MODEL_PARALLEL = False
elif DATASET is None:
    MODEL_FILE = "<UNTRAINED>"
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
