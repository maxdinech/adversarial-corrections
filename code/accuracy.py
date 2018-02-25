"""
Computes the Top-k error of a trained model, over a given subset of a dataset.

---

usage: python3 accuracy.py [-k K] model dataset subset

positional arguments:
  model       Trained model to evaluate
  dataset     Dataset used for training
  subset      Subset to calculate the error on

optional arguments:
  -split SPLIT  Images proportion in val (default: 1/6)
  -k K          Top-k error metric (default: 1)
"""


import sys
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from basics import to_Var, load_model
import data_loader


# Parameters parsing
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Trained model to evaluate")
parser.add_argument("dataset", type=str, help="Dataset used for training")
parser.add_argument("subset", type=str, help="Subset to compute the error on")
parser.add_argument("-split", type=str, default="1/6",
                    help="Images proportion in val (default: 1/6)")
parser.add_argument("-k", type=int, default=1,
                    help="Top-k error metric (default: 1)")
args = parser.parse_args()

model_name = args.model
dset_name = args.dataset
subset = args.subset
val_split = eval(args.split)  # Allows to pass fractions in parameters
k = args.k


# Loads the model
model = load_model(dset_name, model_name)


# Loads the specified subset from the dataset.
images, labels = getattr(data_loader, subset)(dset_name, val_split)


# Custom progress bar.
def bar(data):
    bar_format = "{percentage:3.0f}% |{bar}| {elapsed} - ETA:{remaining}"
    return tqdm(data, ncols=74, bar_format=bar_format)


# Computes the Top-k acccuracy of the model.
# (computing the accuracy mini-batch after mini-batch avoids memory overload)
def accuracy(images, labels, k=1):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=100, shuffle=False)
    count = 0
    position = 0
    for (x, y) in bar(loader):
        # print(f"{position:5}/{len(images)}", end='\r')
        position += len(x)
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        y_pred_k = y_pred.topk(k, 1, True, True)[1]
        count += sum(sum((y_pred_k.t() == y).float())).data
        # .double(): ByteTensor sums are limited at 256.
    return 100 * count / len(images)


# Prints the losses and accuracies at the end of each epoch.
print(f"Computing the Top-{k} error on the {len(images)} {subset} images.")
acc = accuracy(images, labels, k)
error = 100 - acc
print(f"Top-{k} error: {error:0.2f}%")
