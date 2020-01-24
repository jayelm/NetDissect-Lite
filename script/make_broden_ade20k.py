"""
ade20k
"""

import pandas as pd
from operator import itemgetter
import numpy as np

index = pd.read_csv('./dataset/broden1_224/index.csv')

dsets = index.image.str.split('/').map(itemgetter(0))

ade = index[dsets == 'ade20k'].copy()

ade.to_csv('./dataset/broden1_224/index_ade20k.csv', index=False, float_format='%d')

# Random version: shuffle all masks

ade['color'] = np.random.permutation(ade['color'].values)
ade['material'] = np.random.permutation(ade['material'].values)
ade['texture'] = np.random.permutation(ade['texture'].values)
ade['object'] = np.random.permutation(ade['object'].values)
ade['part'] = np.random.permutation(ade['part'].values)
ade['scene'] = np.random.permutation(ade['scene'].values)

ade.to_csv('./dataset/broden1_224/index_ade20k_random.csv', index=False, float_format='%d')
