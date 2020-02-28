"""
ops -> Object Part Scene
"""

import pandas as pd
from operator import itemgetter
from collections import Counter

index = pd.read_csv("./dataset/broden1_224/index.csv")

ops = index[index.material.isnull() & index.texture.isnull()]

dsets = ops.image.str.split("/").map(itemgetter(0))
print(Counter(dsets))

ops.to_csv("./dataset/broden1_224/index_ops.csv", index=False)
