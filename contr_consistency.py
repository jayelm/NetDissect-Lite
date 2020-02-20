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


BY = 'feat_corr'


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

    records = []
    prev_tally = None
    for i, (ln, cs) in enumerate(zip(layernames, contrs_spread)):
        if ln == 'final':
            # Do it manually
            n_classes = cs['weight']['weight'].shape[0]
            for unit in range(n_classes):
                contrs = np.where(cs[BY]['contr'][0][unit])[0]
                contr_labs = [prev_tally[c + 1]['label'] for c in contrs]
                contr_c = summary.pairwise_sim_l(contr_labs)
                records.append({
                    'layer': ln,
                    'unit': unit + 1,
                    'iou': None,
                    'label': ade20k.I2S[unit],
                    'contr_c': contr_c,
                    'contr': ';'.join(contr_labs)
                })
        else:
            ln_full = os.path.join(settings.OUTPUT_FOLDER, f'tally_{ln}.csv')
            tally_df = pd.read_csv(ln_full)
            tally = {d['unit']: d for d in tally_df.to_dict('records')}

            for row_i, row in tally_df.iterrows():
                lab = row['label']
                # 0 index
                u = row['unit'] - 1
                if i == 0:
                    # No contributions - consistency all 1
                    contr_c = 1.0
                    contr_labs = ''
                else:
                    contrs = np.where(cs[BY]['contr'][0][u])[0]
                    contr_labs = [prev_tally[c + 1]['label'] for c in contrs]
                    contr_c = summary.pairwise_sim_l(contr_labs)

                records.append({
                    'layer': ln,
                    'unit': row['unit'],
                    'iou': row['score'],
                    'label': row['label'],
                    'contr_c': contr_c,
                    'contr': ';'.join(contr_labs)
                })

            prev_tally = tally

    contr_df = pd.DataFrame(records)
    contr_df.to_csv(os.path.join(settings.OUTPUT_FOLDER, 'contr_c.csv'), index=False)
