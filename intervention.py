"""
Run an image one-off
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import settings
from imageio import imread
from util.misc import safe_layername
import os
import pickle
from tqdm import tqdm
import pandas as pd

from loader.data_loader import ade20k
from loader.data_loader.broden import normalize_image
from loader.model_loader import loadmodel
from dissection.neuron import NeuronOperator, hook_feature

from torch.utils.data import TensorDataset, DataLoader
from util.train_utils import AverageMeter


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    #  parser.add_argument('imgs', nargs='+', help='Path to image(s)')
    #  parser.add_argument('--query', nargs='*', help='Classes to report stats for')

    args = parser.parse_args()

    feature_names = ["layer4"]
    layernames = list(map(safe_layername, feature_names))

    # Rerun but only capture last layer (layer4)
    model = loadmodel(hook_feature, feature_names=feature_names)

    fo = NeuronOperator()

    features, maxfeature, preds, logits = fo.feature_extraction(
        model=model, feature_names=feature_names, features_only=True
    )

    # Keep last layer only
    features = features[-1]
    maxfeature = maxfeature[-1]
    thresholds = fo.quantile_threshold(
        features, savepath=f"quantile_{layernames[-1]}.npy"
    )

    # Load contributors
    with open(os.path.join(settings.OUTPUT_FOLDER, "contrib.pkl"), "rb") as f:
        contrs = pickle.load(f)
    # Last layer (contributors)
    contrs = contrs[-1]

    # Sanity check - if I do average pooling and feed this into the fully
    # connected layer, I get out the original logits.
    features_dset = TensorDataset(torch.from_numpy(features))
    features_loader = DataLoader(
        features_dset, batch_size=32, shuffle=False, pin_memory=True, num_workers=0
    )
    features_loader_s = DataLoader(
        features_dset, batch_size=32, shuffle=True, pin_memory=True, num_workers=0
    )

    logits_recon = []
    for (x,) in tqdm(features_loader, desc="Sanity check"):
        if settings.GPU:
            x = x.cuda()

        with torch.no_grad():
            x_mean = x.mean(2).mean(2)
            lr = model.fc(x_mean)

        logits_recon.append(lr.cpu().numpy())

    logits_recon = np.concatenate(logits_recon, 0)

    np.testing.assert_allclose(logits, logits_recon, atol=1e-5)

    BY = "weight"
    contrs = contrs[BY]["contr"][0]
    cl2contr = {c: np.where(contrs[c])[0] for c in range(settings.NUM_CLASSES)}

    c_acc = {
        "preint": {c: AverageMeter() for c in cl2contr.keys()},
    }
    ALPHAS = [1, 5, 10]
    for alpha in ALPHAS:
        c_acc[f"int-{alpha}"] = {c: AverageMeter() for c in cl2contr.keys()}
    for c, contr in tqdm(cl2contr.items(), total=len(cl2contr)):
        c_thresh = thresholds[contr].astype(np.float32)
        N_BATCHES = 5
        for (x_preint,), _ in zip(features_loader_s, range(N_BATCHES)):
            for mode in c_acc.keys():
                if mode == "preint":
                    x = x_preint
                else:
                    # Get alpha
                    alpha = int(mode.split("-")[1])
                    x = x_preint.clone()
                    x[:, contr] = torch.from_numpy(
                        c_thresh[:, np.newaxis, np.newaxis] + alpha
                    )

                if settings.GPU:
                    x = x.cuda()

                with torch.no_grad():
                    x_mean = x.mean(2).mean(2)
                    lr = model.fc(x_mean)

                pred = lr.argmax(1)
                acc = (pred == c).float().mean().item()
                c_acc[mode][c].update(acc, x.shape[0])

    all_c_records = []
    for mode in c_acc.keys():
        c_acc_m = {c: m.avg for c, m in c_acc[mode].items()}
        c_records = [
            {"class": c, "label": ade20k.I2S[c], "acc": acc, "mode": mode}
            for c, acc in c_acc_m.items()
        ]
        all_c_records.extend(c_records)
    pd.DataFrame(all_c_records).to_csv(
        os.path.join(settings.OUTPUT_FOLDER, "int_acc.csv"), index=False
    )
