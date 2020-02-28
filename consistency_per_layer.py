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

from bs4 import BeautifulSoup


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="consistency per layer",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output_folder", default=None)

    args = parser.parse_args()

    if args.output_folder is not None:
        output_f = args.output_folder
    else:
        output_f = settings.OUTPUT_FOLDER

    layernames = list(map(safe_layername, settings.FEATURE_NAMES))

    records = []
    for ln in layernames:
        html_fname = os.path.join(output_f, "html", f"{ln}.html")
        with open(html_fname, "r") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        unit_soup = soup.find_all("div", "unit")

        for us in unit_soup:
            unit = us.find("span", "unitnum").text.strip().split("unit ")[1]
            iou = us.find("span", "iou").text.strip().split("IoU ")[1]
            label = us.find("div", "unitlabel").text.strip()
            # consistency
            label, consistency = label.split(" (consistency: ")
            consistency = consistency[:-1]

            records.append(
                {
                    "layer": ln,
                    "label": label,
                    "consistency": float(consistency),
                    "unit": int(unit),
                    "iou": float(iou),
                }
            )

    pd.DataFrame(records).to_csv(os.path.join(output_f, "consistency.csv"), index=False)
