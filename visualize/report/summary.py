"""
Trying to get summaries of neurons
"""

import spacy
import loader.data_loader.formula as F
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm


nlp = spacy.load('en_vectors_web_lg')


def summarize(f, namer):
    leaves = f.get_vals()
    leaves = [namer(l) for l in leaves]
    # Remove scene suffixes
    leaves = [l[:-2] if l.endswith('-s') else l for l in leaves]
    leaves = [l for l in leaves if l in nlp.vocab]
    if any(l not in nlp.vocab for l in leaves):
        print([l for l in leaves if l not in nlp.vocab])
    vecs = [nlp(l).vector for l in leaves]
    vec = np.array(vecs).mean(0)[np.newaxis]
    keys, _, sims, = nlp.vocab.vectors.most_similar(vec, n=1, batch_size=10000)
    key = keys.item()
    sim = sims.item()

    best_word = nlp.vocab.strings[key]
    return best_word, sim
