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
import shutil


def generate_final_layer_summary(ds, weight, last_features, last_thresholds, last_preds, last_logits, prev_layername=None, prev_tally=None, contributors=None):
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    ed.ensure_dir('html', 'image', 'final')

    html_fname = ed.filename(f"html/{expdir.fn_safe('final')}.html")

    print(f"Generating html summary {html_fname}")
    html = [html_common.HTML_PREFIX]
    html.append(f'<h3>{settings.OUTPUT_FOLDER} - Final Layer</h3>')

    # Loop through classes
    # Last layer of contributors
    contributors = contributors[-1]
    for cl in range(weight.shape[0]):
        # TODO: Make this compatible with non-ade20k
        cl_name = ade20k.I2S[cl]
        html.append(
            f'<div class="card contr contr_final">'
                f'<div class="card-header">'
                    f'<h5 class="mb-0">{cl_name}</h5>'
                f'</div>'
                f'<div class="card-body">'
        )
        all_contrs = []
        for contr_i, contr_name in enumerate(sorted(list(contributors.keys()))):
            contr_dict = contributors[contr_name]
            if contr_dict['contr'][0] is None:
                continue
            contr, inhib = contr_dict['contr']
            contr = np.where(contr[cl])[0]
            contr_weights = contr_dict['weight'][cl, contr]
            all_contrs.extend(contr)

            inhib = np.where(inhib[cl])[0]
            inhib_weights = contr_dict['weight'][cl, inhib]

            contr_url_str = ','.join(map(str, contr))
            contr_labels = [f'{u + 1} ({prev_tally.get(u + 1, "unk")}, {w:.3f})' for u, w in
                            sorted(zip(contr, contr_weights), key=lambda x: x[1], reverse=True)]
            contr_labels = [f'<span class="label contr-label" data-unit="{u + 1}" data-clname="{cl_name}">{l}</span>' for u, l in zip(contr, contr_labels)]

            inhib_url_str = ','.join(map(str, inhib))
            inhib_labels = [f'{u + 1} ({prev_tally.get(u + 1, "unk")}, {w:.3f})' for u, w in
                            sorted(zip(inhib, inhib_weights), key=lambda x: x[1], reverse=True)]
            inhib_labels = [f'<span class="label inhib-label" data-unit="{u + 1}" data-clname="{cl_name}">{l}</span>' for u, l in zip(inhib, inhib_labels)]

            contr_label_str = ', '.join(contr_labels)
            inhib_label_str = ', '.join(inhib_labels)

            html.append(
                f'<p class="contributors"><a href="{prev_layername}.html?u={contr_url_str}">Contributors ({contr_name}): {contr_label_str}</a></p>'
                f'<p class="inhibitors"><a href="{prev_layername}.html?u={inhib_url_str}">Inhibitors ({contr_name}): {inhib_label_str}</a></p>'
            )

        all_contrs = list(set(all_contrs))
        # Save images with highest logits
        cl_images = [i for i in range(len(last_features)) if f"{ds.scene(i)}-s" == cl_name]
        cl_images = sorted(cl_images, key=lambda i: last_logits[i, cl], reverse=True)
        if cl_images:
            for i, im_index in enumerate(cl_images[:5]):
                #  breakpoint()
                imfn = ds.filename(im_index)
                imfn_base = os.path.basename(imfn)
                html_imfn = ed.filename(f"html/image/final/{imfn_base}")
                shutil.copy(imfn, html_imfn)
                html.append(
                    #  f'<canvas class="mask-canvas" id="{cl_name}-{i}" width="100" height="100" data-src="">Your browser does not support canvas</canvas>"'
                    f'<img loading="lazy" class="final-img" id="{cl_name}-{i}" data-clname="{cl_name}" width="100" height="100" data-imfn="{imfn_base}" src="image/final/{imfn_base}">'
                )
                # Save masks
                for unit in all_contrs:
                    imfn_alpha = imfn_base.replace('.jpg', '.png')
                    #  breakpoint()
                    feats = last_features[im_index, unit]
                    thresh = last_thresholds[unit]
                    mask = (feats > thresh).astype(np.uint8) * 255
                    mask = np.clip(mask, 50, 255)
                    mask = Image.fromarray(mask).resize((settings.IMG_SIZE, settings.IMG_SIZE), resample=Image.BILINEAR)
                    # All black
                    mask_alpha = Image.fromarray(np.zeros((settings.IMG_SIZE, settings.IMG_SIZE), dtype=np.uint8), mode='L')
                    mask_alpha.putalpha(mask)
                    mask_fname = ed.filename(f"html/image/final/mask-{unit}-{imfn_alpha}")
                    mask_alpha.save(mask_fname)
                    # Upscale mask...save asa

        html.append(f'</div></div>')

    html.append(html_common.HTML_SUFFIX);
    with open(html_fname, 'w') as f:
        f.write('\n'.join(html))
