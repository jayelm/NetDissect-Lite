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
import settings
import numpy as np
from PIL import Image
import warnings
from tqdm import tqdm
import loader.data_loader.formula as F
import os


import pycocotools.mask as cmask
# unit,category,label,score

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]


MG = .5
BLUE_TINT = np.array([-MG * 255, -MG * 255, MG * 255])
RED_TINT = np.array([MG * 255, -MG * 255, -MG * 255])

def add_colored_masks(img, feat_mask, unit_mask):
    img = img.astype(np.int64)

    nowhere_else = np.logical_not(np.logical_or(feat_mask, unit_mask)).astype(np.int64)
    nowhere_else = (nowhere_else * 0.8 * 255).astype(np.int64)
    nowhere_else = nowhere_else[:, :, np.newaxis]

    feat_mask = feat_mask[:, :, np.newaxis] * BLUE_TINT
    feat_mask = np.round(feat_mask).astype(np.int64)

    img += feat_mask

    unit_mask = unit_mask[:, :, np.newaxis] * RED_TINT
    unit_mask = np.round(unit_mask).astype(np.int64)

    img += unit_mask

    img -= nowhere_else

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s



def generate_html_summary(ds, layer, preds, mc, maxfeature=None, features=None, thresholds=None,
        imsize=None, imscale=72, tally_result=None,
        contributors=None, prev_layername=None, prev_tally=None,
        gridwidth=None, gap=3, limit=None, force=False, verbose=False):
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
    html.append(html_common.get_sortheader(['score', 'unit']))
    html.append('</div>')

    if gridwidth is None:
        gridname = ''
        gridwidth = settings.TOPN
        gridheight = 1
    else:
        gridname = '-%d' % gridwidth
        gridheight = (settings.TOPN + gridwidth - 1) // gridwidth

    html.append('<div class="unitgrid"') # Leave off > to eat spaces
    if limit is not None:
        rendered_order = rendered_order[:limit]
    for i, record in enumerate(
            sorted(rendered_order, key=lambda record: -float(record['score']))):
        record['score-order'] = i
    for label_order, record in enumerate(tqdm(rendered_order, desc='Visualizing neurons')):
        unit = int(record['unit']) - 1 # zero-based unit indexing
        imfn = 'image/%s%s-%04d.jpg' % (
                expdir.fn_safe(layer), gridname, unit)

        # Compute 2nd and 3rd image metadata
        lab_f = F.parse(record['label'], reverse_namer=ds.rev_name)
        if settings.EMBEDDING_SUMMARY and len(lab_f) > 1:
            # Add a summary; load spacy only if needed
            from visualize.report import summary
            summ, sim = summary.summarize(lab_f, lambda j: ds.name(None, j))
            summ = f" ({summ} {sim:.3f})"
        else:
            summ = ''

        # Minor negation of label
        neglab_f = F.minor_negate(lab_f, hard=True)
        neglab = neglab_f.to_str(lambda name: ds.name(None, name))

        row2fn = 'image/%s%s-%04d-maskimg.jpg' % (expdir.fn_safe(layer), gridname, unit)
        row3fn = 'image/%s%s-%04d-maskimg-neg1.jpg' % (expdir.fn_safe(layer), gridname, unit)

        if force or not ed.has('html/%s' % imfn):
            if verbose:
                print('Visualizing %s unit %d' % (layer, unit))
            # ==== ROW 1: TOP PATCH IMAGES ====
            img_ann = []
            for index in top[unit]:
                if settings.PROBE_DATASET == 'cub':
                    # Images can be different in CUB - stick with Image.open for more reliable
                    # operation
                    image = np.array(Image.open(ds.filename(index)).convert('RGB'))
                else:
                    image = imread(ds.filename(index))
                mask = np.array(Image.fromarray(features[index][unit]).resize(image.shape[:2], resample=Image.BILINEAR))
                mask = mask > thresholds[unit]
                if settings.PROBE_DATASET == 'cub':
                    mask = mask.T  # Needs to be transposed, not sure why
                vis = (mask[:, :, numpy.newaxis] * 0.8 + 0.2) * image
                if settings.PROBE_DATASET == 'cub':
                    vis = vis.round().astype(np.uint8)
                if vis.shape[:2] != (imsize, imsize):
                    vis = np.array(Image.fromarray(vis).resize((imsize, imsize), resample=Image.BILINEAR))
                vis = np.round(vis).astype(np.uint8)
                img_ann.append({
                    'img': vis,
                    'labels': [None],
                    'mark': None
                })
            tiled = create_tiled_image(img_ann, gridheight, gridwidth, ds, imsize=imsize, gap=gap)
            imwrite(ed.filename('html/' + imfn), tiled)

            # ==== ROW 2 - other images that match the mask ====
            labs_enc = mc.get_mask(lab_f)
            labs = cmask.decode(labs_enc)
            # Unflatten
            labs = labs.reshape((features.shape[0], *mc.mask_shape))
            # sum up
            lab_tallies = labs.sum((1, 2))
            # get biggest tallies
            idx = np.argsort(lab_tallies)[::-1][:settings.TOPN]
            mask_imgs_ann = []
            for i in idx:
                fname = ds.filename(i)
                img = np.array(Image.open(fname))
                # FEAT MASK: blue
                feat_mask = np.array(Image.fromarray(labs[i]).resize(img.shape[:2]))

                # UNIT MASK: red
                unit_mask = np.array(Image.fromarray(features[i][unit]).resize(img.shape[:2], resample=Image.BILINEAR))
                unit_mask = unit_mask > thresholds[unit]

                intersection = np.logical_and(feat_mask, unit_mask).sum()
                union = np.logical_or(feat_mask, unit_mask).sum()
                iou = intersection / (union + 1e-10)
                lbl = f"{iou:.3f}"

                img_masked = add_colored_masks(img, feat_mask, unit_mask)

                mask_imgs_ann.append({
                    'img': img_masked,
                    'labels': [lbl],
                    'mark': None
                })
            tiled = create_tiled_image(mask_imgs_ann, gridheight, gridwidth, ds, imsize=imsize, gap=gap)
            imwrite(ed.filename('html/' + row2fn), tiled)

            # ==== ROW 3 - images thatt match slightly neegative masks ====
            labs_enc = mc.get_mask(neglab_f)
            labs = cmask.decode(labs_enc)
            # Unflatten
            labs = labs.reshape((features.shape[0], *mc.mask_shape))
            # Sum up
            lab_tallies = labs.sum((1, 2))
            # Get biggest tallies
            idx = np.argsort(lab_tallies)[::-1][:settings.TOPN]

            mask_imgs_ann = []
            for i in idx:
                fname = ds.filename(i)
                img = np.array(Image.open(fname))
                # FEAT MASK: blue
                feat_mask = np.array(Image.fromarray(labs[i]).resize(img.shape[:2]))

                # UNIT MASK: red
                unit_mask = np.array(Image.fromarray(features[i][unit]).resize(img.shape[:2], resample=Image.BILINEAR))
                unit_mask = unit_mask > thresholds[unit]

                intersection = np.logical_and(feat_mask, unit_mask).sum()
                union = np.logical_or(feat_mask, unit_mask).sum()
                iou = intersection / (union + 1e-10)
                lbl = f"{iou:.3f}"

                img_masked = add_colored_masks(img, feat_mask, unit_mask)

                mask_imgs_ann.append({
                    'img': img_masked,
                    'labels': [lbl],
                    'mark': None
                })

            tiled = create_tiled_image(mask_imgs_ann, gridheight, gridwidth, ds, imsize=imsize, gap=gap)
            imwrite(ed.filename('html/' + row3fn), tiled)

        # Get neighbors
        contrs = []
        for contr_i, contr_name in enumerate(sorted(list(contributors.keys()))):
            contr_dict = contributors[contr_name]
            if contr_dict['contr'][0] is None:
                continue
            contr, inhib = contr_dict['contr']

            contr = np.where(contr[unit])[0]
            contr_label_str = ', '.join(f'{u} ({prev_tally.get(u, "unk")})' for u in contr)
            contr_url_str = ','.join(map(str, contr))

            inhib = np.where(inhib[unit])[0]
            inhib_url_str = ','.join(map(str, inhib))
            inhib_label_str = ', '.join(f'{u} ({prev_tally.get(u, "unk")})' for u in inhib)

            show = 'show' if contr_i == 0 else ''

            cname = f"{contr_name}-{unit}"
            cstr = (
                f'<div class="contr-head card-header" id="heading-{cname}"><h5 class="mb-0"><button class="btn btn-link" data-toggle="collapse" data-target="#collapse-{cname}" aria-expanded="true" aria-controls="collapse-{cname}">{contr_name}</button></h5></div>'
                f'<div id="collapse-{cname}" class="collapse {show}" aria-labelledby="heading-{cname}" data-parent="#contr-{unit}"><div class="card-body">'
                    f'<div class="card-body">'
                    f'<p class="contributors"><a href="{prev_layername}.html?u={contr_url_str}">Contributors: {contr_label_str}</a></p>'
                    f'<p class="inhibitors"><a href="{prev_layername}.html?u={inhib_url_str}">Inhibitors: {inhib_label_str}</a></p>'
                    f'</div>'
                f'</div></div>'
            )
            cstr = f'<div class="card contr">{cstr}</div>'
            contrs.append(cstr)
        contr_str = '\n'.join(contrs)
        contr_str = f'<div id="contr-{unit}">{contr_str}</div>'

        # Generate the wrapper HTML
        graytext = ' lowscore' if float(record['score']) < settings.SCORE_THRESHOLD else ''
        html.append('><div class="unit%s" data-order="%d %d %d">' %
                (graytext, label_order, record['score-order'], unit + 1))
        html.append(f"<div class='unitlabel'>{fix(record['label'])}{summ}</div>")
        html.append('<div class="info">' +
            '<span class="layername">%s</span> ' % layer +
            '<span class="unitnum">unit %d</span> ' % (unit + 1) +
            '<span class="category">(%s)</span> ' % record['category'] +
            '<span class="iou">IoU %.2f</span>' % float(record['score']) +
            contr_str +
            '</div>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (imfn, imscale))
        html.append('<p class="midrule">Other examples of feature (<span class="bluespan">feature mask</span> <span class="redspan">unit mask</span>)</p>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (row2fn, imscale))
        html.append(f'<p class="midrule">Examples of {neglab}</p>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (row3fn, imscale))
        html.append('</div') # Leave off > to eat spaces
    html.append('></div>')
    html.extend([html_common.HTML_SUFFIX]);
    with open(ed.filename('html/%s.html' % expdir.fn_safe(layer)), 'w') as f:
        f.write('\n'.join(html))
