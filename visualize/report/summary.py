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


def clean(leaves, to_underscore=False):
    if to_underscore:
        leaves = [l.replace('-', '_') for l in leaves]
        leaves = [l[:-2] if l.endswith('_s') else l for l in leaves]
        leaves = [l[:-2] if l.endswith('_c') else l for l in leaves]
    else:
        leaves = [l.replace('-', ' ').replace('_', ' ') for l in leaves]
        leaves = [l[:-2] if l.endswith(' s') else l for l in leaves]
        leaves = [l[:-2] if l.endswith(' c') else l for l in leaves]
    # Remove scene suffixes
    return leaves


def get_nlps(f, namer, vals=False):
    """
    Get spacy nlp objects for leaves in formula, with some cleanup
    """
    if not vals:
        leaves = f.get_vals()
    else:
        leaves = f
    leaves = [namer(l).lower() for l in leaves]
    leaves = clean(leaves)
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


PATH_TYPES = [
    'hypernyms',
    'instance_hypernyms',
    'hyponyms',
    'instance_hyponyms',
    'member_meronyms',
    'part_meronyms',
    'substance_meronyms',
    'similar_tos',
    'also_sees',
]


def find_path(src, dest, path_types=PATH_TYPES):
    """
    Find path between wordnet synsets
    """
    visited = set()

    # Create a queue for BFS
    queue = []

    queue.append((src, tuple()))
    visited.add(src)

    while queue:
        # Dequeue a vertex from queue
        n, path = queue.pop(0)

        # If this adjacent node is the destination node,
        # then return true
        if n == dest:
             return path

        #  Else, continue to do BFS
        relations = set()
        for path_type in path_types:
            relations.update(getattr(n, path_type)())
        for i in relations:
            if i not in visited:
                new_path = path + (n, )
                queue.append((i, new_path))
                visited.add(i)

    return None


def wn_midpoint(src, dest):
    path = find_path(src, dest)
    if path is None:
        return None
    midp = len(path) // 2
    return path[midp]


def wn_summarize(f, namer):
    """
    Get one-word summary of label which is closest in embedding space
    (TODO: have option to have it NOT be any of the labels)
    """
    leaves = f.get_vals()
    leaves = [namer(l).lower() for l in leaves]
    leaves = clean(leaves, to_underscore=True)

    synsets = [get_synset(s) for s in leaves]
    if all(s is None for s in synsets):
        return 'unk', 0.0
    synsets = [s for s in synsets if s is not None]

    if len(synsets) == 1:
        return str(synsets[0])
    # Find midpoint between each pair, then find midpoints of the midpoints
    while len(synsets) > 1:
        a = synsets.pop(0)
        b = synsets.pop(0)
        midp = wn_midpoint(a, b)
        if midp is None:
            return 'unk', 0.0
        synsets.append(midp)

    return str(synsets[0])
