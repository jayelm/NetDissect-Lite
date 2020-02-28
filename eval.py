"""
Run an image one-off
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import settings
from imageio import imread

from loader.data_loader import ade20k
from loader.data_loader.broden import normalize_image
from loader.model_loader import loadmodel

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("imgs", nargs="+", help="Path to image(s)")
    parser.add_argument("--query", nargs="*", help="Classes to report stats for")

    args = parser.parse_args()

    model = loadmodel(None)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    query_i = torch.tensor([ade20k.S2I[q] for q in args.query])

    for img_fname in args.imgs:
        #  breakpoint()
        img = imread(img_fname)
        #  img = Image.open(img_fname)

        #  img = transform(img)
        img = normalize_image(img, bgr_mean=[109.5388, 118.6897, 124.6901])[np.newaxis]

        # Some reversal/normalization stuff?
        img = torch.from_numpy(img[:, ::-1, :, :].copy())
        img.div_(255.0 * 0.224)

        #  img_t = img.unsqueeze(0)
        if settings.GPU:
            img = img.cuda()

        preds = model(img)

        #  breakpoint()
        pred = preds[0].argmax().item()
        score = preds[0].max().item()

        pred_lbl = ade20k.I2S[pred]

        pred_queries = preds[0, query_i]
        query_str = ", ".join(
            f"{q}: {qs.item():.3f}" for q, qs in zip(args.query, pred_queries)
        )
        query_str = f"({query_str})"

        print(f"{img_fname}\t{pred_lbl}: {score:.3f}, queries: {query_str}")
