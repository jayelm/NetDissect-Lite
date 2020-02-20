"""
Compute consistency measures for contributors
"""

from visualize.report import summary
from dissection.neuron import NeuronOperator
from util.misc import safe_layername
import settings
import pickle
import os
import pandas as pd
import numpy as np
from loader.data_loader import ade20k


def contr_c(layernames, contrs_spread):
    """
    Get dataframe which has stats of contributors
    """
    records = []
    prev_tally = None
    for i, (ln, cs) in enumerate(zip(layernames, contrs_spread)):
        for by in cs.keys():
            if ln == 'final':
                # Do it manually
                n_classes = cs['weight']['weight'].shape[0]
                for unit in range(n_classes):
                    contrs = np.where(cs[by]['contr'][0][unit])[0]
                    contr_labs = [prev_tally[c + 1]['label'] for c in contrs]
                    contr_c = summary.pairwise_sim_l(contr_labs)
                    records.append({
                        'layer': ln,
                        'unit': unit + 1,
                        'iou': None,
                        'label': ade20k.I2S[unit],
                        'contr_c': contr_c,
                        'contr': ';'.join(contr_labs),
                        'n_contr': len(contr_labs),
                        'by': by
                    })
            else:
                ln_full = os.path.join(settings.OUTPUT_FOLDER, f'tally_{ln}.csv')
                tally_df = pd.read_csv(ln_full)
                tally = {d['unit']: d for d in tally_df.to_dict('records')}

                for row_i, row in tally_df.iterrows():
                    # 0 index
                    u = row['unit'] - 1
                    if i == 0:
                        # No contributions - consistency all 1
                        contr_labs = []
                        contr_c = 1.0
                    else:
                        contrs = np.where(cs[by]['contr'][0][u])[0]
                        contr_labs = [prev_tally[c + 1]['label'] for c in contrs]
                        contr_c = summary.pairwise_sim_l(contr_labs)

                    records.append({
                        'layer': ln,
                        'unit': row['unit'],
                        'iou': row['score'],
                        'label': row['label'],
                        'contr_c': contr_c,
                        'contr': ';'.join(contr_labs),
                        'n_contr': len(contr_labs),
                        'by': by
                    })

        prev_tally = tally

    return pd.DataFrame(records)


def contr_final(layernames, contrs_spread):
    """
    Get dataframe which is just all contributors (+ scores) for the final layer
    """
    cs = contrs_spread[-1]
    n_classes = cs['weight']['weight'].shape[0]
    records = []

    ln_full = os.path.join(settings.OUTPUT_FOLDER, f'tally_{layernames[-2]}.csv')
    tally_df = pd.read_csv(ln_full)
    prev_tally = {d['unit']: d for d in tally_df.to_dict('records')}

    for by in cs.keys():
        for cl in range(n_classes):
            weights = cs[by]['weight'][cl]
            contrs = cs[by]['contr'][0][cl]
            for ui, (w, c) in enumerate(zip(weights, contrs)):
                clab = prev_tally[ui + 1]['label']
                records.append({
                    'class': cl + 1,
                    'label': ade20k.I2S[cl],
                    'unit': ui + 1,
                    'contr_label': clab,
                    'weight': w,
                    'is_contr': c,
                    'by': by
                })
    return pd.DataFrame(records)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='__doc__',
        formatter_class=ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    fo = NeuronOperator()
    data = fo.data

    layernames = list(map(safe_layername, settings.FEATURE_NAMES + ['final']))
    with open(os.path.join(settings.OUTPUT_FOLDER, 'contrib.pkl'), 'rb') as f:
        contrs_spread = pickle.load(f)

    contr_c_df = contr_c(layernames, contrs_spread)
    contr_c_df.to_csv(os.path.join(settings.OUTPUT_FOLDER, 'contr_c.csv'), index=False)

    contr_final_df = contr_final(layernames, contrs_spread)
    contr_final_df.to_csv(os.path.join(settings.OUTPUT_FOLDER, 'contr_final.csv'), index=False)
