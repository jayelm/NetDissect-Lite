"""
Compute average consistency per layer

Because I'm too lazy to restructure to save consistency with the original
tallies (though I should probably do that eventually...)
"""

import re
import settings
from util.misc import safe_layername
import os
import pandas as pd
import numpy as np


CONSISTENCY_REGEX = re.compile(r'consistency: (\d*\.?\d+)')


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='consistency per layer',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_folder', default=None)

    args = parser.parse_args()

    if args.output_folder is not None:
        output_f = args.output_folder
    else:
        output_f = settings.OUTPUT_FOLDER

    layernames = list(map(safe_layername, settings.FEATURE_NAMES))

    records = []
    for ln in layernames:
        html_fname = os.path.join(output_f, 'html', f'{ln}.html')
        with open(html_fname, 'r') as f:
            html = f.read()

        matches = re.findall(CONSISTENCY_REGEX, html)
        consistency = list(map(float, matches))
        consistency = np.array(consistency)

        for c in consistency:
            records.append({
                'layer': ln,
                'c': c
            })

    pd.DataFrame(records).to_csv(os.path.join(output_f, 'consistency.csv'), index=False)
