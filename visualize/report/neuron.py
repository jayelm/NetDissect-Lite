'''
viewprobe creates visualizations for a certain eval.
'''

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
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
    html = [html_prefix]
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
    html.append(html_sortheader)
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
    for label_order, record in enumerate(tqdm(rendered_order, desc='Images')):
        unit = int(record['unit']) - 1 # zero-based unit indexing
        imfn = 'image/%s%s-%04d.jpg' % (
                expdir.fn_safe(layer), gridname, unit)
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
            lab_f = F.parse(record['label'], reverse_namer=ds.rev_name)
            labs_enc = mc.get_mask(lab_f)
            if settings.EMBEDDING_SUMMARY and len(lab_f) > 1:
                # Add a summary; load spacy only if needed
                from visualize.report import summary
                summ, sim = summary.summarize(lab_f, lambda j: ds.name(None, j))
                summ = f" ({summ} {sim:.3f})"
            else:
                summ = ''
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
            row2fn = 'image/%s%s-%04d-maskimg.jpg' % (expdir.fn_safe(layer), gridname, unit)
            tiled = create_tiled_image(mask_imgs_ann, gridheight, gridwidth, ds, imsize=imsize, gap=gap)
            imwrite(ed.filename('html/' + row2fn), tiled)

            # ==== ROW 3 - images thatt match slightly neegative masks ====
            neglab_f = F.minor_negate(lab_f, hard=True)
            neglab = neglab_f.to_str(lambda name: ds.name(None, name))
            labs_enc = mc.get_mask(neglab_f)
            labs = cmask.decode(labs_enc)
            # Unflatten
            labs = labs.reshape((features.shape[0], *mc.mask_shape))
            # Sum up
            lab_tallies = labs.sum((1, 2))
            # Get biggest tallies
            idx = np.argsort(lab_tallies)[::-1][:settings.TOPN]

            row3fn = 'image/%s%s-%04d-maskimg-neg1.jpg' % (expdir.fn_safe(layer), gridname, unit)
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
    html.extend([html_suffix]);
    with open(ed.filename('html/%s.html' % expdir.fn_safe(layer)), 'w') as f:
        f.write('\n'.join(html))

html_prefix = '''
<!doctype html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
<style>
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .layername {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 288px;
  height: 72px;
}
.bluespan {
    color: blue;
}
.redspan {
    color: red;
}
.midrule {
    margin-top: 1em;
    margin-bottom: 0.25em;
}
.unit {
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img-wrapper {
  text-align: center;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  text-decoration: none;
  font-weight: bold;
  cursor: default;
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
'''

html_sortheader = '''
<div class="sortheader">
sort by
<span class="sortby currentsort" data-index="0">label</span>
<span class="sortby" data-index="1">score</span>
<span class="sortby" data-index="2">unit</span>
</div>
'''

html_suffix = '''
</div>
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid">
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>
<script>
$('img:not([data-nothumb])[src]').wrap(function() {
  var result = $('<a data-toggle="lightbox">')
  result.attr('href', $(this).attr('src'));
  var caption = $(this).closest('figure').find('figcaption').text();
  if (!caption && $(this).closest('.citation').length) {
    caption = $(this).closest('.citation').text();
  }
  if (caption) {
    result.attr('data-footer', caption);
  }
  var title = $(this).attr('title');
  if (!title) {
    title = $(this).closest('td').find('.unit,.score').map(function() {
      return $(this).text(); }).toArray().join('; ');
  }
  if (title) {
    result.attr('data-title', title);
  }
  return result;
});
$(document).on('click', '[data-toggle=lightbox]', function(event) {
    $('#lightbox img').attr('src', $(this).attr('href'));
    $('#lightbox .modal-title').text($(this).data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').text($(this).data('footer') ||
       $(this).closest('.unit').find('.info').text());
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('keydown', function(event) {
    $('#lightbox').modal('hide');
});
$(document).on('click', '.sortby', function(event) {
    var sortindex = +$(this).data('index');
    sortBy(sortindex);
    $('.sortby').removeClass('currentsort');
    $(this).addClass('currentsort');
});
function sortBy(index) {
  $('.unitgrid').find('.unit').sort(function (a, b) {
     return +$(a).eq(0).data('order').split(' ')[index] -
            +$(b).eq(0).data('order').split(' ')[index];
  }).appendTo('.unitgrid');
}
</script>
</body>
</html>
'''
