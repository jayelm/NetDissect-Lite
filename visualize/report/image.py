import settings
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from imageio import imread

def create_tiled_image(imgs, gridheight, gridwidth, ds, imsize=112, gap=3):
    tiled = np.full(
        ((imsize + gap) * gridheight - gap,
         (imsize + gap) * gridwidth - gap, 3), 255, dtype='uint8')
    # TODO: Make imgs a dict
    for x, (img, label, mark) in enumerate(imgs):
        row = x // gridwidth
        col = x % gridwidth
        if not isinstance(img, np.ndarray):
            if settings.PROBE_DATASET == 'cub':
                # Images can be different in CUB - stick with Image.open for more reliable
                # operation
                vis = Image.open(ds.filename(img)).convert('RGB')
            else:
                vis = Image.fromarray(imread(ds.filename(img)))
            if vis.size[:2] != (imsize, imsize):
                vis = vis.resize((imsize, imsize), resample=Image.BILINEAR)
        else:
            vis = Image.fromarray(img)

        if mark is not None:
            vis = ImageOps.expand(vis, border=10, fill=mark).resize((imsize, imsize), resample=Image.BILINEAR)
        if label is not None:
            draw = ImageDraw.Draw(vis)
            draw.text((0, 0), label)
        vis = np.array(vis)
        tiled[row*(imsize+gap):row*(imsize+gap)+imsize,
              col*(imsize+gap):col*(imsize+gap)+imsize,:] = vis
    return tiled
