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
from tqdm import tqdm


from visualize.report import summary
import posthoc


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="consistency per layer",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output_folder", default=None)
    parser.add_argument("--feature_names", nargs="+", default=None)

    args = parser.parse_args()

    if args.output_folder is not None:
        output_f = args.output_folder
    else:
        output_f = settings.OUTPUT_FOLDER

    if args.feature_names is not None:
        layernames = list(map(safe_layername, args.feature_names))
    else:
        layernames = list(map(safe_layername, settings.FEATURE_NAMES))

    for ln in layernames:
        print(ln)
        records = []
        html_fname = os.path.join(output_f, "html", f"{ln}.html")

        soup = posthoc.load_html(html_fname)
        i = 0
        for record, unit in tqdm(
            posthoc.units(soup, layername=ln, yield_soup=True), desc=ln
        ):
            i += 1
            if i > 10:
                break
            wns, wns_sim = summary.wn_summarize(record["label"], lambda x: x)
            emb, emb_sim = summary.emb_summarize(record["label"], lambda x: x)
            u = unit.find("div", "unitlabel")
            u.string = f"{u.text} (wn summary: {wns}) (emb summary: {emb})"

            # We don't need the original label anymore
            record["wn_summary"] = wns
            record["wn_summary_sim"] = wns_sim
            record["emb_summary"] = emb
            record["emb_summary_sim"] = emb_sim
            record["label"] = record["label_str"]
            del record["label_str"]

            records.append(record)

        wn_summarized = html_fname.replace(".html", "_summarized.html")
        with open(wn_summarized, "w") as f:
            f.write(str(soup))

        csv_fname = os.path.join(output_f, f"tally_{ln}_summarized.csv")
        pd.DataFrame(records).to_csv(csv_fname, index=False)
