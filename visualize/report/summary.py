"""
Trying to get summaries of neurons
"""

import spacy
import itertools
import numpy as np
from nltk.corpus import wordnet as wn
import functools


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
    leaves = [l[:-2] if l.endswith(' s') else l for l in leaves]
    leaves = [l[:-2] if l.endswith(' c') else l for l in leaves]
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


def emb_summarize(f, namer, search_n=25):
    """
    Get one-word summary of label which is closest in embedding space
    (TODO: have option to have it NOT be any of the labels)
    """
    nlps = get_nlps(f, namer)
    nlps = filter_oov(nlps)
    vecs = [n.vector for n in nlps]

    toks_flat = set()
    for n in nlps:
        toks_flat.update(list(n))
    toks_flat = [t.text for t in toks_flat]

    vec = np.array(vecs).mean(0)[np.newaxis]
    keys, _, sims, = nlp.vocab.vectors.most_similar(
        vec, n=search_n, batch_size=10000
    )
    keys = keys[0]
    sims = sims[0]

    for k, s in zip(keys, sims):
        w = nlp.vocab.strings[k].lower()
        if w not in toks_flat:
            return w, s

    # Just return the original word, but mark that it's not the same member
    w = nlp.vocab.strings[keys[0]].lower()
    return f'{w}-same', sims[0]


def get_synset(t):
    ss = wn.synsets(t, pos=wn.NOUN)
    if ss:
        return ss[0]
    ss = wn.synsets(t.split('_')[0], pos=wn.NOUN)
    if ss:
        return ss[0]
    else:
        return None


def wn_summarize(f, namer):
    """
    Get one-word summary of label which is closest in embedding space
    (TODO: have option to have it NOT be any of the labels)
    """
    leaves = f.get_vals()
    leaves = [namer(l).lower() for l in leaves]
    leaves = [l.replace('-', ' ').replace('_', ' ') for l in leaves]
    # Remove scene suffixes
    leaves = [l[:-2] if l.endswith('-s') else l for l in leaves]

    ss = []
    for l in leaves:
        s = get_synset(l)
        if s is None:
            return 'unk', 0.0
        ss.append(s)
    the_ss = functools.reduce(lambda a, b: a.lowest_common_hypernyms(b)[0], ss)

    return str(the_ss)
