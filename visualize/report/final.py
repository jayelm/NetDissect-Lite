'''
viewprobe creates visualizations for a certain eval.
'''

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
from visualize.report import html_common
from visualize.report.image import create_tiled_image, score_histogram
from loader.data_loader import ade20k
import settings
import numpy as np
from PIL import Image
import warnings
from tqdm import tqdm
import loader.data_loader.formula as F
import os


def generate_final_layer_summary(ds, weight, prev_layername=None, prev_tally=None, contributors=None):
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)

    html_fname = ed.filename(f"html/{expdir.fn_safe('final')}.html")

    print(f"Generating html summary {html_fname}")
    html = [html_common.HTML_PREFIX]
    html.append(f'<h3>{settings.OUTPUT_FOLDER} - Final Layer</h3>')

    # Loop through classes
    for cl in range(weight.shape[0]):

        # TODO: Make this compatible with non-ade20k
        cl_name = ade20k.I2S[cl]

        contr = np.where(contributors[0][cl])[0]
        contr_weights = weight[cl, contr]
        inhib = np.where(contributors[1][cl])[0]
        inhib_weights = weight[cl, inhib]

        contr_url_str = ','.join(map(str, contr))
        contr_label_str = ', '.join(f'{u} ({prev_tally.get(u, "unk")}, {w:.3f})' for u, w in zip(contr, contr_weights))

        inhib_url_str = ','.join(map(str, inhib))
        inhib_label_str = ', '.join(f'{u} ({prev_tally.get(u, "unk")}, {w:.3f})' for u, w in zip(inhib, inhib_weights))

        cstr = (
            f'<div class="card contr contr_final">'
                f'<div class="card-header">'
                    f'<h5 class="mb-0">{cl_name}</h5>'
                f'</div>'
                f'<div class="card-body">'
                    f'<p class="contributors"><a href="{prev_layername}.html?u={contr_url_str}">Contributors: {contr_label_str}</a></p>'
                    f'<p class="inhibitors"><a href="{prev_layername}.html?u={inhib_url_str}">Inhibitors: {inhib_label_str}</a></p>'
                f'</div>'
            f'</div>'
        )
        html.append(cstr)

    html.append(html_common.HTML_SUFFIX);
    with open(html_fname, 'w') as f:
        f.write('\n'.join(html))
