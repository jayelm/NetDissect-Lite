"""
For each class, load images and save as numpy arrays.
"""

import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

from scipy.spatial import KDTree

PARTS = [
    'back',
    'beak',
    'belly',
    'breast',
    'crown',
    'forehead',
    'left eye',
    'left leg',
    'left wing',
    'nape',
    'right eye',
    'right leg',
    'right wing',
    'tail',
    'throat',
]


def segment_class(args):
    seg_name, seg_part_ids, seg_part_locs, metadata = args
    seg_arr = np.array(Image.open(seg_name))
    kdt = KDTree(seg_part_locs)
    # Binarize
    seg_arr = (seg_arr >= 128).astype(np.uint8)
    indexes = np.argwhere(seg_arr)
    for ind in indexes:
        # Get nearest neighbor
        nearest_point = kdt.query(ind)[1]
        nearest_part_id = seg_part_ids[nearest_point]
        seg_arr[ind[0], ind[1]] = nearest_part_id

    return seg_arr, metadata


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Save numpy',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cub_dir', default='dataset/CUB_200_2011',
                        help='Directory to load/cache')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of multiprocessing workers')

    args = parser.parse_args()

    img_dir = os.path.join(args.cub_dir, 'images')
    bird_classes = os.listdir(img_dir)

    os.makedirs(os.path.join(args.cub_dir, 'parts', 'segmentations'),
                exist_ok=True)

    # Load parts
    images = pd.read_csv(os.path.join(args.cub_dir, 'images.txt'),
                         sep=' ', names=['image_id', 'image_name'])
    images['image_id'] = images['image_id'] - 1

    part_locs = pd.read_csv(os.path.join(args.cub_dir, 'parts', 'part_locs.txt'),
                            sep=' ',
                            names=['image_id', 'part_id', 'x', 'y', 'visible'])
    part_locs['image_id'] = part_locs['image_id'] - 1
    part_locs['part_id'] = part_locs['part_id'] - 1

    part_locs = part_locs[part_locs['visible'] == 1]
    assert len(part_locs['image_id'].unique() == 11788)

    def mp_args_generator():
        for i, row in images.iterrows():
            img_name = row['image_name']
            bc = img_name.split('/')[0]
            seg_name = os.path.join(args.cub_dir, 'segmentations', img_name)
            seg_name = seg_name.replace('.jpg', '.png')
            seg_parts = part_locs[part_locs['image_id'] == row['image_id']]
            seg_part_ids = seg_parts['part_id'].to_numpy()
            seg_part_locs = np.array(list(zip(seg_parts['x'], seg_parts['y'])))
            metadata = (bc, img_name)
            yield (seg_name, seg_part_ids, seg_part_locs, metadata)

    arrs = {bc: {} for bc in bird_classes}
    gt = {bc: np.load(os.path.join(args.cub_dir, 'images', bc, 'img.npz')) for
          bc in bird_classes}
    with mp.Pool(args.workers) as pool:
        with tqdm(total=images.shape[0], desc='Classes') as pbar:
            for (seg_arr, (bc, img_name)) in pool.imap_unordered(segment_class, mp_args_generator()):
                arrs[bc][img_name] = seg_arr
                assert bc in gt and img_name in gt[bc]
                pbar.update()

    for cl, cl_arrs in tqdm(arrs.items(), total=len(arrs), desc='Saving'):
        fname = os.path.join(args.cub_dir, 'parts', 'segmentations', f'{cl}.npz')
        np.savez_compressed(fname, cl_arrs)
