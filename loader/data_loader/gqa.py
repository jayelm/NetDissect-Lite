from functools import partial
import numpy
import os
import re
import random
import signal
import csv
import settings
import numpy as np
from collections import OrderedDict, defaultdict
from imageio import imread
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import pandas as pd
from PIL import Image, ImageDraw

try:
    import ujson as json
except ImportError:
    import json

from . import data_utils as du


def load_gqa(data_dir, max_images=None):
    """
    Load GQA dataset.
    """
    return {"train": GQADataset(data_dir, max_images=max_images, split="val")}


def to_bbox(odata):
    return [odata["x"], odata["y"], odata["x"] + odata["w"], odata["y"] + odata["h"]]


def annotate_image(im, graph):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    for odata in graph["objects"].values():
        bb = to_bbox(odata)
        draw.rectangle(bb, fill=None, outline="black")
        draw.text((odata["x"], odata["y"]), odata["name"])
    return im


class GQADataset:

    DIM = 224

    def __init__(self, data_dir, max_images=None, split="train"):
        self.data_dir = data_dir
        self.split = split
        sg_fname = os.path.join(data_dir, f"{split}_sceneGraphs.json")
        with open(sg_fname, "r") as f:
            self.scene_graphs = json.load(f)
        self.scene_ids = list(self.scene_graphs.keys())
        if max_images is not None:
            self.scene_ids = self.scene_ids[:max_images]

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, i):
        sid = self.scene_ids[i]
        graph = self.scene_graphs[sid]
        img_fname = os.path.join(self.data_dir, "images", f"{sid}.jpg")
        img = Image.open(img_fname)
        img_rsz = img.resize((self.DIM, self.DIM))
        img_arr = np.array(img_rsz)
        graph_rsz = self.resize_graph(graph)
        return img_arr, graph_rsz

    def resize_graph(self, graph):
        # TODO: Cache resized values?
        w_ratio = self.DIM / graph["width"]
        h_ratio = self.DIM / graph["height"]
        new_graph = {"width": self.DIM, "height": self.DIM, "objects": {}}
        for oid, odata in graph["objects"].items():
            new_odata = odata.copy()
            new_odata["h"] = int(round(odata["h"] * h_ratio))
            new_odata["w"] = int(round(odata["w"] * w_ratio))
            new_odata["y"] = int(round(odata["y"] * h_ratio))
            new_odata["x"] = int(round(odata["x"] * w_ratio))
            new_graph["objects"][oid] = new_odata
        return new_graph
