'''
viewprobe creates visualizations for a certain eval.
'''

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
from visualize.report import html_common, cards
from visualize.report.image import create_tiled_image, score_histogram
import settings
import numpy as np
from PIL import Image
import warnings
from tqdm import tqdm
import loader.data_loader.formula as F
from loader.data_loader import ade20k
import os
import shutil


import pycocotools.mask as cmask
# unit,category,label,score


def generate_html_summary(ds, layer, preds, mc, maxfeature=None, features=None, thresholds=None,
        imsize=None, imscale=72, tally_result=None,
        contributors=None, prev_layername=None, prev_tally=None, prev_features=None, prev_thresholds=None,
        gridwidth=None, gap=3, limit=None, force=False, verbose=False, skip=False):
    if skip:
        return
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    print('Generating html summary %s' % ed.filename('html/%s.html' % expdir.fn_safe(layer)))
    if verbose:
        print('Sorting units by score.')
    if imsize is None:
        imsize = settings.IMG_SIZE
    # 512 units x top_n.
    # For each unit, images that get the highest activation (anywhere?), in
    # descending order.
    top = np.argsort(maxfeature, 0)[:-1 - settings.TOPN:-1, :].transpose()
    ed.ensure_dir('html','image')
    html = [html_common.HTML_PREFIX]
    html.append(f'<h3>{settings.OUTPUT_FOLDER}</h3>')
    rendered_order = []
    barfn = 'image/%s-bargraph.svg' % (
            expdir.fn_safe(layer))
    try:
        bargraph.bar_graph_svg(ed, layer,
                               tally_result=tally_result,
                               rendered_order=rendered_order,
                               save=ed.filename('html/' + barfn))
    except ValueError as e:
        # Probably empty
        warnings.warn(f"could not make svg bargraph: {e}")
        pass
    html.extend([
        '<div class="histogram">',
        '<img class="img-fluid" src="%s" title="Summary of %s %s">' % (
            barfn, ed.basename(), layer),
        '</div>'
        ])
    # histogram 2 ==== iou
    ioufn = f'image/{expdir.fn_safe(layer)}-iou.svg'
    ious = [float(r['score']) for r in rendered_order]
    iou_mean = np.mean(ious)
    iou_std = np.std(ious)
    iou_title = f'IoUs ({iou_mean:.3f} +/- {iou_std:.3f})'
    score_histogram(rendered_order, os.path.join(ed.directory, 'html', ioufn),
                    title=iou_title)
    html.extend([
        '<div class="histogram">',
        '<img class="img-fluid" src="%s" title="Summary of %s %s">' % (
            ioufn, ed.basename(), layer),
        '</div>'
        ])
    html.append(html_common.FILTERBOX)
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    html.append('%d/%d units covering %d concepts with IoU &ge; %.2f' % (
        len([record for record in rendered_order
            if float(record['score']) >= settings.SCORE_THRESHOLD]),
        len(rendered_order),
        len(set(record['label'] for record in rendered_order
            if float(record['score']) >= settings.SCORE_THRESHOLD)),
        settings.SCORE_THRESHOLD))
    html.append('</div>')
    sort_by = ['score', 'unit']
    if settings.SEMANTIC_CONSISTENCY:
        sort_by.append('consistency')
    html.append(html_common.get_sortheader(sort_by))
    html.append('</div>')

    if gridwidth is None:
        gridname = ''
        gridwidth = settings.TOPN
        gridheight = 1
    else:
        gridname = '-%d' % gridwidth
        gridheight = (settings.TOPN + gridwidth - 1) // gridwidth

    html.append('<div class="unitgrid">')
    if limit is not None:
        rendered_order = rendered_order[:limit]

    # Assign ordering based on score
    for i, record in enumerate(
            sorted(rendered_order, key=lambda record: -float(record['score']))):
        record['score-order'] = i

    # Assign ordering based on consistency
    for record in rendered_order:
        lab_f = F.parse(record['label'], reverse_namer=ds.rev_name)
        if settings.SEMANTIC_CONSISTENCY:
            from visualize.report import summary
            record['consistency'] = summary.pairwise_sim(lab_f, lambda j: ds.name(None, j))
        else:
            record['consistency'] = 0
    for i, record in enumerate(
            sorted(rendered_order, key=lambda record: -float(record['consistency']))):
        record['consistency-order'] = i

    # TODO: Make embedding summary searchable too.

    # Visualize neurons
    card_htmls = {}
    for label_order, record in enumerate(tqdm(rendered_order, desc='Visualizing neurons')):
        card_html = cards.make_card_html(
            ed, label_order, record, ds, mc, layer, gridname, top, features, thresholds,
            preds, contributors, prev_layername, prev_features, prev_thresholds, prev_tally,
            force=force
        )
        html.append(card_html)
        card_htmls[record['unit']] = card_html

    html.append('</div>')
    html.extend([html_common.HTML_SUFFIX]);
    with open(ed.filename('html/%s.html' % expdir.fn_safe(layer)), 'w') as f:
        f.write('\n'.join(html))

    return card_htmls
