"""
Split broden into its consistuent datasets
"""

import pandas as pd
from operator import itemgetter
import numpy as np
np.random.seed(0)

index = pd.read_csv('./dataset/broden1_224/index.csv')

dsets = index.image.str.split('/').map(itemgetter(0))

SPLITS = dsets.unique()

for split in SPLITS:
    print(split)
    splitdf = index[dsets == split].copy()

    splitdf.to_csv(f'./dataset/broden1_224/index_{split}.csv', index=False, float_format='%d')

    # Random version: shuffle all masks
    splitdf['color'] = np.random.permutation(splitdf['color'].values)
    splitdf['material'] = np.random.permutation(splitdf['material'].values)
    splitdf['texture'] = np.random.permutation(splitdf['texture'].values)
    splitdf['object'] = np.random.permutation(splitdf['object'].values)
    splitdf['part'] = np.random.permutation(splitdf['part'].values)
    splitdf['scene'] = np.random.permutation(splitdf['scene'].values)

    splitdf.to_csv(f'./dataset/broden1_224/index_{split}_random.csv', index=False, float_format='%d')
