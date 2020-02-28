"""
Visualize representation-level (neuralese) results
"""

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
from visualize.report import html_common
from visualize.report.image import (
    create_tiled_image,
    score_histogram,
    pairwise_histogram,
)
import settings
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import warnings
import loader.data_loader.formula as F
from collections import Counter
from scipy.stats import percentileofscore
import os

from dissection.representation import ReprOperator as RO, square_to_condensed
from tqdm import tqdm

replacements = [(re.compile(r[0]), r[1]) for r in [(r"-[sc]$", ""), (r"_", " "),]]


def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s


def generate_html_summary(
    ds,
    layer,
    records,
    dist,
    preds,
    mc,
    thresh,
    imsize=None,
    imscale=72,
    gridwidth=None,
    gap=3,
    limit=None,
    force=False,
    verbose=False,
):
    label2img = mc.img2label.T
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    print(
        "Generating html summary %s"
        % ed.filename("html/%s.html" % expdir.fn_safe(layer))
    )
    if verbose:
        print("Sorting units by score.")
    if imsize is None:
        imsize = settings.IMG_SIZE
    # 512 units x top_n.
    # For each unit, images that get the highest activation (anywhere?), in
    # descending order.
    #  top = np.argsort(maxfeature, 0)[:-1 - settings.TOPN:-1, :].transpose()
    r_correct = np.array([r["pred_label"] == r["true_label"] for r in records])
    ed.ensure_dir("html", "image")
    html = [html_common.HTML_PREFIX]
    html.append(f"<h3>{settings.OUTPUT_FOLDER}</h3>")
    html.append(
        f"<p>Accuracy on these examples: {r_correct.sum()}/{len(r_correct)} ({r_correct.mean() * 100:.3f}%)</p>"
    )
    barfn = f"image/{expdir.fn_safe(layer)}-threshold.svg"
    pairwise_histogram(dist, os.path.join(ed.directory, "html", barfn))
    html.extend(
        [
            '<div class="histogram">',
            f"<p>Threshold: {thresh:.3f} (top {settings.REPR_ALPHA * 100}%)</p>",
            '<img class="img-fluid" src="%s" title="Summary of %s %s">'
            % (barfn, ed.basename(), layer),
            "</div>",
        ]
    )
    jacfn = f"image/{expdir.fn_safe(layer)}-jac.svg"
    jacs = [r["score"] for r in records]
    jac_mean = np.mean(jacs)
    jac_std = np.std(jacs)
    jac_title = f"Jaccard distances ({jac_mean:.3f} +/- {jac_std:.3f})"
    score_histogram(records, os.path.join(ed.directory, "html", jacfn), title=jac_title)
    html.extend(
        [
            '<div class="histogram">',
            '<img class="img-fluid" src="%s" title="Summary of %s %s">'
            % (jacfn, ed.basename(), layer),
            "</div>",
        ]
    )
    rendered_order = records
    html.append(html_common.FILTERBOX)
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    html.append(
        "%d/%d images covering %d concepts with jaccard sim &ge; %.2f"
        % (
            len(
                [
                    record
                    for record in rendered_order
                    if float(record["score"]) >= settings.SCORE_THRESHOLD
                ]
            ),
            len(rendered_order),
            len(
                set(
                    record["label"]
                    for record in rendered_order
                    if float(record["score"]) >= settings.SCORE_THRESHOLD
                )
            ),
            settings.SCORE_THRESHOLD,
        )
    )
    html.append("</div>")
    html_sortheader = html_common.get_sortheader(
        ["score", "unit", "correct", "incorrect"]
    )
    html.append(html_sortheader)
    html.append("</div>")

    if gridwidth is None:
        gridname = ""
        gridwidth = settings.TOPN
        gridheight = 1
    else:
        gridname = "-%d" % gridwidth
        gridheight = (settings.TOPN + gridwidth - 1) // gridwidth

    html.append('<div class="unitgrid"')  # Leave off > to eat spaces
    if limit is not None:
        rendered_order = rendered_order[:limit]
    for i, record in enumerate(
        sorted(rendered_order, key=lambda record: -float(record["score"]))
    ):
        record["score-order"] = i
    for i, record in enumerate(
        sorted(
            rendered_order,
            key=lambda record: -int(record["pred_label"] == record["true_label"]),
        )
    ):
        record["correct-order"] = i
    for label_order, record in enumerate(tqdm(rendered_order, desc="Images")):
        inp = int(record["input"])
        imfn = "image/%s%s-%04d.jpg" % (expdir.fn_safe(layer), gridname, inp)
        if force or not ed.has("html/%s" % imfn):
            if verbose:
                print("Visualizing %s unit %d" % (layer, inp))

            # ==== ROW 1 - most similar images ====
            imgs = [
                {"img": inp, "labels": [None, str(record["true_label"])], "mark": 0}
            ]
            # Get the closest images according to distance matrix
            n_img = mc.img2label.shape[0]
            oth_dists = Counter()
            for oth_i in range(n_img):
                if oth_i == inp:
                    continue
                n_oth_i = square_to_condensed(inp, oth_i, n_img)
                oth_dists[oth_i] = -dist[n_oth_i]
            top = oth_dists.most_common(settings.TOPN - 1)

            for x, (index, sim) in enumerate(top):
                if index == inp:
                    continue
                lab = f"{-sim:.2f}"
                mark = (0, 255, 0) if -sim < thresh else (255, 0, 0)
                imgs.append(
                    {"img": index, "labels": [lab, str(preds[index, 1])], "mark": mark}
                )

            tiled = create_tiled_image(
                imgs, gridheight, gridwidth, ds, imsize=imsize, gap=gap
            )
            imwrite(ed.filename("html/" + imfn), tiled)

            # ==== ROW 2 - other images that match the mask ====
            neglab = "<error>"
            row2fn = "image/%s%s-%04d-maskimg.jpg" % (
                expdir.fn_safe(layer),
                gridname,
                inp,
            )
            row3fn = "image/%s%s-%04d-maskimg-neg1.jpg" % (
                expdir.fn_safe(layer),
                gridname,
                inp,
            )
            if (
                isinstance(record["label"], str) and record["label"]
            ):  # Could be empty if no formula found
                lab_f = F.parse(record["label"], reverse_namer=ds.rev_name)
                labs = RO.get_labels(lab_f, labels=label2img)
                if labs.sum() == 0:
                    # Just write an empty image
                    tiled = np.full(
                        (
                            (imsize + gap) * gridheight - gap,
                            (imsize + gap) * gridwidth - gap,
                            3,
                        ),
                        255,
                        dtype="uint8",
                    )
                    imwrite(ed.filename("html/" + row2fn), tiled)
                else:
                    mask_imgs = np.random.choice(
                        np.argwhere(labs).squeeze(1), settings.TOPN + 1
                    )  # Might sample ourselves
                    mask_imgs = [mi for mi in mask_imgs if mi != inp][: settings.TOPN]
                    mask_imgs_ann = []
                    for mi in mask_imgs:
                        n_oth_i = square_to_condensed(inp, mi, n_img)
                        sim = dist[n_oth_i]
                        mark = (0, 255, 0) if sim < thresh else (255, 0, 0)
                        label = f"{sim:.2f}"
                        mask_imgs_ann.append(
                            {
                                "img": mi,
                                "labels": [label, str(preds[mi, 1])],
                                "mark": mark,
                            }
                        )
                    tiled = create_tiled_image(
                        mask_imgs_ann, gridheight, gridwidth, ds, imsize=imsize, gap=gap
                    )
                    imwrite(ed.filename("html/" + row2fn), tiled)

                # ==== ROW 3 - images that match slightly negative masks ====
                negate_attempts = 0
                while True:
                    neglab_f = F.minor_negate(lab_f, hard=negate_attempts == 0)
                    neglab = neglab_f.to_str(lambda name: ds.name(None, name))
                    labs = RO.get_labels(neglab_f, labels=label2img)

                    if labs.sum() == 0:
                        negate_attempts += 1
                        if negate_attempts > 3:
                            # Just write an empty image
                            tiled = np.full(
                                (
                                    (imsize + gap) * gridheight - gap,
                                    (imsize + gap) * gridwidth - gap,
                                    3,
                                ),
                                255,
                                dtype="uint8",
                            )
                            imwrite(ed.filename("html/" + row3fn), tiled)
                            break
                        else:
                            continue

                    mask_imgs = np.random.choice(
                        np.argwhere(labs).squeeze(1), settings.TOPN + 1
                    )  # Might sample ourselves
                    mask_imgs = [mi for mi in mask_imgs if mi != inp][: settings.TOPN]
                    mask_imgs_ann = []
                    for mi in mask_imgs:
                        n_oth_i = square_to_condensed(inp, mi, n_img)
                        sim = dist[n_oth_i]
                        mark = (0, 255, 0) if sim < thresh else (255, 0, 0)
                        label = f"{sim:.2f}"
                        mask_imgs_ann.append(
                            {
                                "img": mi,
                                "labels": [label, str(preds[mi, 1])],
                                "mark": mark,
                            }
                        )
                    tiled = create_tiled_image(
                        mask_imgs_ann, gridheight, gridwidth, ds, imsize=imsize, gap=gap
                    )
                    imwrite(ed.filename("html/" + row3fn), tiled)
                    break

        # Generate the wrapper HTML
        correct = record["pred_label"] == record["true_label"]
        graytext = (
            " lowscore" if float(record["score"]) < settings.SCORE_THRESHOLD else ""
        )
        html.append(
            '><div class="unit%s" data-order="%d %d %d %d %d">'
            % (
                graytext,
                label_order,
                record["score-order"],
                inp,
                record["correct-order"],
                -record["correct-order"],
            )
        )
        html.append('<div class="unitlabel">%s</div>' % fix(record["label"]))
        html.append(
            '<div class="info">'
            + '<span class="layername">%s</span> ' % layer
            + '<span class="unitnum">image %d</span> ' % (inp)
            + '<span class="category">(%s)</span> ' % record["category"]
            + '<span class="iou">Jaccard %.2f</span>' % float(record["score"])
            + f'<span class="prediction-{"correct" if correct else "incorrect"}">pred: {record["pred_label"]} true: {record["true_label"]}</span> '
            + "</div>"
        )
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' % (imfn, imscale)
        )
        html.append('<p class="midrule">Other examples of feature</p>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>'
            % (row2fn, imscale)
        )
        html.append(f'<p class="midrule">Examples of {neglab}</p>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>'
            % (row3fn, imscale)
        )
        html.append("</div")  # Leave off > to eat spaces
    html.append("></div>")
    html.extend([html_common.HTML_SUFFIX])
    with open(ed.filename("html/%s.html" % expdir.fn_safe(layer)), "w") as f:
        f.write("\n".join(html))
