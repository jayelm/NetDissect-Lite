'''
viewprobe creates visualizations for a certain eval.
'''

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
import settings
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import warnings
import loader.data_loader.formula as F
from collections import Counter
from scipy.stats import percentileofscore
import seaborn as sns
import os

from repr_operation import FeatureOperator as FO, square_to_condensed
# unit,category,label,score

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s


def histogram(dist, fname, n=10000):
    # Take a sample of the distances
    samp_n = np.random.randint(len(dist), size=n)
    samp = dist[samp_n]
    plt = sns.distplot(samp)
    ax = plt.get_figure()
    ax.savefig(fname)


def generate_html_summary(ds, layer, records, dist, mc, thresh,
        imsize=None, imscale=72,
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
    #  top = np.argsort(maxfeature, 0)[:-1 - settings.TOPN:-1, :].transpose()
    ed.ensure_dir('html','image')
    html = [html_prefix]
    barfn = f'image/{expdir.fn_safe(layer)}-threshold.svg'
    histogram(dist, os.path.join(ed.directory, 'html', barfn))
    html.extend([
        '<div class="histogram">',
        f'<p>Threshold: {thresh:.3f} (top {settings.REPR_ALPHA * 100}%%)</p>',
        '<img class="img-fluid" src="%s" title="Summary of %s %s">' % (
            barfn, ed.basename(), layer),
        '</div>'
        ])
    rendered_order = records
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    html.append('%d/%d images covering %d concepts with jaccard sim &ge; %.2f' % (
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
    for label_order, record in enumerate(rendered_order):
        inp = int(record['input'])
        imfn = 'image/%s%s-%04d.jpg' % (
                expdir.fn_safe(layer), gridname, inp)
        if force or not ed.has('html/%s' % imfn):
            if verbose:
                print('Visualizing %s unit %d' % (layer, inp))
            # Visualize this image, retrieving it from
            this_img_fname = ds.filename(inp)
            # Generate the top-patch image
            tiled = numpy.full(
                ((imsize + gap) * gridheight - gap,
                 (imsize + gap) * gridwidth - gap, 3), 255, dtype='uint8')
            # Visualize the current image.
            vis = Image.fromarray(imread(this_img_fname))
            if vis.size[:2] != (imsize, imsize):
                vis = vis.resize((imsize, imsize), resample=Image.BILINEAR)
            # Add a red border
            vis = ImageOps.expand(vis, border=10).resize((imsize, imsize), resample=Image.BILINEAR)
            vis = np.array(vis)
            tiled[0*(imsize+gap):0*(imsize+gap)+imsize,
                  0*(imsize+gap):0*(imsize+gap)+imsize,:] = vis

            # Get the closest images according to distance matrix
            n_img = mc.img2label.shape[0]
            oth_dists = Counter()
            for oth_i in range(n_img):
                if oth_i == inp:
                    continue
                n_oth_i = square_to_condensed(inp, oth_i, n_img)
                oth_dists[oth_i] = -dist[n_oth_i]

            top = oth_dists.most_common(settings.TOPN)

            # TODO: Need to store "most similar" images so can
            # So we can loop through the most similar.
            for x, (index, sim) in enumerate(top):
                if x == 0:
                    continue  # skip the first
                row = x // gridwidth
                col = x % gridwidth
                if settings.PROBE_DATASET == 'cub':
                    # Images can be different in CUB - stick with Image.open for more reliable
                    # operation
                    vis = np.array(Image.open(ds.filename(index)).convert('RGB'))
                else:
                    vis = imread(ds.filename(index))
                if vis.shape[:2] != (imsize, imsize):
                    vis = np.array(Image.fromarray(vis).resize((imsize, imsize), resample=Image.BILINEAR))
                vis = Image.fromarray(vis)
                draw = ImageDraw.Draw(vis)
                # Label w/ similarity and similarity percentile
                label = f"{-sim:.1f}"
                draw.text((0, 0), label)
                vis = np.array(vis)
                tiled[row*(imsize+gap):row*(imsize+gap)+imsize,
                      col*(imsize+gap):col*(imsize+gap)+imsize,:] = vis

            imwrite(ed.filename('html/' + imfn), tiled)
        # Generate the wrapper HTML
        graytext = ' lowscore' if float(record['score']) < settings.SCORE_THRESHOLD else ''
        html.append('><div class="unit%s" data-order="%d %d %d">' %
                (graytext, label_order, record['score-order'], inp + 1))
        html.append('<div class="unitlabel">%s</div>' % fix(record['label']))
        html.append('<div class="info">' +
            '<span class="layername">%s</span> ' % layer +
            '<span class="unitnum">image %d</span> ' % (inp + 1) +
            '<span class="category">(%s)</span> ' % record['category'] +
            '<span class="iou">Jaccard %.2f</span>' % float(record['score']) +
            '</div>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (imfn, imscale))
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
