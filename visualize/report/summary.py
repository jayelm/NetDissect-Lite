"""
Trying to get summaries of neurons
"""

import spacy
import itertools
import numpy as np


print("Loading spacy...")
nlp = spacy.load('en_vectors_web_lg')
print("done")


def filter_oov(nlps):
    return [n for n in nlps if not (n.vector == 0.0).all()]


def get_nlps(f, namer, vals=False):
    """
    Get spacy nlp objects for leaves in formula, with some cleanup
    """
    if not vals:
        leaves = f.get_vals()
    else:
        leaves = f
    leaves = [namer(l).lower() for l in leaves]
    leaves = [l.replace('-', ' ').replace('_', ' ') for l in leaves]
    # Remove scene suffixes
    leaves = [l[:-2] if l.endswith('-s') else l for l in leaves]
    return [nlp(l) for l in leaves]


def pairwise_sim_l(vals, namer=lambda x: x):
    """
    Compute average pairwise similarity between a list of atomic labels
    """
    if len(vals) <= 1:
        return 1.0
    nlps = get_nlps(vals, namer, vals=True)
    nlps = filter_oov(nlps)
    if not nlps:
        return 0  # out of vocab
    sims = []
    for v1, v2 in itertools.combinations(nlps, 2):
        sims.append(v1.similarity(v2))
    return np.mean(np.array(sims))


def pairwise_sim(f, namer):
    """
    Compute average pairwise similarity between formulas
    Compute pairwise similarity between formulas, averaged
    """
    if len(f) <= 1:
        return 1.0
    nlps = get_nlps(f, namer)
    nlps = filter_oov(nlps)
    if not nlps:
        return 0  # out of vocab
    sims = []
    for v1, v2 in itertools.combinations(nlps, 2):
        sims.append(v1.similarity(v2))
    return np.mean(np.array(sims))


def summarize(f, namer):
    """
    Get one-word summary of label which is closest in embedding space
    (TODO: have option to have it NOT be any of the labels)
    """
    nlps = get_nlps(f, namer)
    nlps = filter_oov(nlps)
    vecs = [n.vector for n in nlps]
    vec = np.array(vecs).mean(0)[np.newaxis]
    keys, _, sims, = nlp.vocab.vectors.most_similar(vec, n=1, batch_size=10000)
    key = keys.item()
    sim = sims.item()

    best_word = nlp.vocab.strings[key].lower()
    return best_word, sim
