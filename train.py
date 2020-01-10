"""
Train a model for CUB classification
"""

import settings
from loader.model_loader import loadmodel
from loader.data_loader.cub import load_cub, to_dataloader
from tqdm import tqdm, trange
import numpy as np
from collections import defaultdict
import contextlib
import warnings

import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from util import train_utils as tutil


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', default='dataset/CUB_200_2011',
                        help='Dataset to load from')
    parser.add_argument('--batch_size', default=32,
                        type=int, help='Train batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Training epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Default seed')
    parser.add_argument('--save_dir', default='./zoo/trained/resnet18_cub_finetune',
                        help='Where to save the model')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if settings.DATASET != 'imagenet':
        warnings.warn('Recommend imagenet-pretained model')

    torch.manual_seed(args.seed)
    random = np.random.seed(args.seed)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tutil.save_args(args, args.save_dir)

    datasets = load_cub(args.data_dir, random_state=random,
                        max_classes=5 if args.debug else None)
    dataloaders = {s: to_dataloader(d, batch_size=args.batch_size)
                   for s, d in datasets.items()}

    model = loadmodel(None)
    # Replace the last layer
    model.fc = nn.Linear(in_features=512, out_features=datasets['train'].n_classes)
    # Re-move the model on/off GPU
    if settings.GPU:
        model = model.cuda()
    else:
        model = model.cpu()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    def run(split, epoch):
        training = split == 'train'
        loader = dataloaders[split]
        meters = {m: tutil.AverageMeter() for m in ['loss', 'acc']}
        if training:
            model.train()
            ctx = contextlib.nullcontext()
        else:
            model.eval()
            ctx = torch.no_grad()

        progress_loader = tqdm(loader)
        with ctx:
            for batch_i, (imgs, classes, *_) in enumerate(progress_loader):
                if settings.GPU:
                    imgs = imgs.cuda()
                    classes = classes.cuda()

                batch_size = imgs.shape[0]
                if training:
                    optimizer.zero_grad()

                logits = model(imgs)
                loss = criterion(logits, classes)

                if training:
                    loss.backward()
                    optimizer.step()

                preds = logits.argmax(1)
                acc = (preds == classes).float().mean()
                meters['loss'].update(loss.item(), batch_size)
                meters['acc'].update(acc.item(), batch_size)

                progress_loader.set_description(f"{split.upper():<6} {epoch:3} loss {meters['loss'].avg:.4f} acc {meters['acc'].avg:.4f}")

        return {k: m.avg for k, m in meters.items()}

    metrics = defaultdict(list)
    metrics['best_val_acc'] = 0.0
    metrics['best_val_loss'] = float('inf')
    metrics['best_epoch'] = 0
    for epoch in trange(args.epochs, desc='Epoch'):
        for split in ['train', 'val', 'test']:
            split_metrics = run(split, epoch)
            for m, val in split_metrics.items():
                metrics[f'{split}_{m}'].append(val)
        tqdm.write('')

        if metrics['val_acc'][-1] > metrics['best_val_acc']:
            metrics['best_val_acc'] = metrics['val_acc'][-1]
            metrics['best_val_loss'] = metrics['val_loss'][-1]
            metrics['best_epoch'] = epoch
            tutil.save_model(model, True, args.save_dir)

        tutil.save_metrics(metrics, args.save_dir)
