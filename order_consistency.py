"""
Get highest consistency
"""


import pandas as pd
import settings
import os
from util.misc import safe_layername
from collections import defaultdict

from loader.data_loader import ade20k

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--output_folder", default=None)

    args = parser.parse_args()

    if args.output_folder is not None:
        output_f = args.output_folder
    else:
        output_f = settings.OUTPUT_FOLDER

    cst = pd.read_csv(os.path.join(output_f, "consistency.csv"))
    contr_final = pd.read_csv(os.path.join(output_f, "contr_final.csv"))
    BY = "feat_corr"

    contr_final = contr_final[(contr_final["by"] == BY) & contr_final["is_contr"]]
    contr_dicts = contr_final.to_dict("records")
    contr2cl = defaultdict(list)
    cl2contr = defaultdict(list)
    for record in contr_dicts:
        cl2contr[record["class"]].append(record["unit"])
        contr2cl[record["unit"]].append(record["class"])
    #  contr2cl = dict(contr2cl)
    #  cl2contr = dict(cl2contr)

    def get_cls(unit):
        cl = contr2cl[unit]
        cl_label = [ade20k.I2S[c - 1] for c in cl]
        return ",".join(cl_label)

    layernames = [safe_layername(ln) for ln in settings.FEATURE_NAMES]

    cst = cst[cst["layer"] == layernames[-1]]

    cst["classes"] = cst["unit"].apply(get_cls)

    does_contribute = cst[cst["classes"] != ""]
    does_contribute = does_contribute.sort_values(by="consistency")

    does_contribute.to_csv(os.path.join(output_f, "final2classes.csv"), index=False)
