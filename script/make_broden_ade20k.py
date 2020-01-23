"""
ade20k
"""

import pandas as pd
from operator import itemgetter

index = pd.read_csv('./dataset/broden1_224/index.csv')

dsets = index.image.str.split('/').map(itemgetter(0))

ade = index[dsets == 'ade20k']

ade.to_csv('./dataset/broden1_224/index_ade20k.csv', index=False, float_format='%d')
