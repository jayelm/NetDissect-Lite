"""
For each class, load images and save as numpy arrays.
"""

import numpy as np
import os
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Save numpy',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cub_dir', default='dataset/CUB_200_2011',
                        help='Directory to load/cache')

    args = parser.parse_args()

    img_dir = os.path.join(args.cub_dir, 'images')
    for bird_class in tqdm(os.listdir(img_dir), desc='Classes'):
        bird_imgs_np = {}
        class_dir = os.path.join(img_dir, bird_class)
        bird_imgs = sorted([x for x in os.listdir(class_dir) if x != 'img.npz'])
        for bird_img in bird_imgs:
            bird_img_fname = os.path.join(class_dir, bird_img)
            img = Image.open(bird_img_fname).convert('RGB')
            img_np = np.asarray(img)

            bird_imgs_np[os.path.join(bird_class, bird_img)] = img_np

        np_fname = os.path.join(class_dir, 'img.npz')
        np.savez_compressed(np_fname, **bird_imgs_np)
