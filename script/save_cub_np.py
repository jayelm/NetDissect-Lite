"""
For each class, load images and save as numpy arrays.
"""

import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp


def npify(args):
    img_dir, bird_class = args
    bird_imgs_np = {}
    class_dir = os.path.join(img_dir, bird_class)
    bird_imgs = sorted([x for x in os.listdir(class_dir) if x != "img.npz"])
    for bird_img in bird_imgs:
        bird_img_fname = os.path.join(class_dir, bird_img)
        img = Image.open(bird_img_fname).convert("RGB")
        img_np = np.asarray(img)

        bird_imgs_np[os.path.join(bird_class, bird_img)] = img_np

    np_fname = os.path.join(class_dir, "img.npz")
    np.savez_compressed(np_fname, **bird_imgs_np)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Save numpy", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cub_dir", default="dataset/CUB_200_2011", help="Directory to load/cache"
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of multiprocessing workers"
    )

    args = parser.parse_args()

    img_dir = os.path.join(args.cub_dir, "images")
    bird_classes = os.listdir(img_dir)

    mp_args = [(img_dir, bc) for bc in bird_classes]
    with mp.Pool(args.workers) as pool:
        with tqdm(total=len(mp_args), desc="Classes") as pbar:
            for _ in pool.imap_unordered(npify, mp_args):
                pbar.update()
