import spacy
import loader.data_loader.formula as F


nlp = spacy.load('en_vectors_web_lg')


def summarize(f):
    breakpoint()
    leaves = F.get_leaves(f)
    # Remove scene suffixes
    leaves = [l[:-2] if l.endswith('-s') else l for l in leaves]
    vecs = [nlp(l).vector for l in leaves]
    return 'foo'
