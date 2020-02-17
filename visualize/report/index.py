'''
Create index page of the entire network
'''

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
from visualize.report import html_common
from visualize.report.image import create_tiled_image, score_histogram
from loader.data_loader import ade20k
from visualize.report import tree
import settings
import numpy as np
from PIL import Image
import warnings
from tqdm import tqdm
import loader.data_loader.formula as F
import os
import shutil
import json


def generate_index(layernames, contrs, tallies):
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    ed.ensure_dir('html', 'image')

    html_fname = ed.filename(f"html/{expdir.fn_safe('index')}.html")

    print(f"Generating html index {html_fname}")
    html = [html_common.HTML_PREFIX]

    for ln in layernames:
        html.append(
            f'<a href="{expdir.fn_safe(ln)}.html"><h5>{ln}</h5></a>'
        )

    html.append(
        f'<a href="final.html"><h5>Final decisions</h5></a>'
    )

    tree_data = tree.make_treedata(contrs, tallies, units=range(1, 365, 10),
                                   maxchildren=3, maxdepth=4)
    tree_data_str = json.dumps(tree_data)
    tree_data_str = f"<script>var treeData = {tree_data_str};</script>"
    html.append(tree_data_str)
    html.append(tree.TREESTYLE)
    html.append(tree.TREESCRIPT)
    html.append(html_common.HTML_SUFFIX);

    with open(html_fname, 'w') as f:
        f.write('\n'.join(html))
