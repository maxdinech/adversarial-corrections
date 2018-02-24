"""
Evaluates the accuracy of a trained model, over a given subset of a dataset.

---

usage: python3 accuracy.py [-t T] model dataset subset

positional arguments:
  model       Trained model to evaluate
  dataset     Dataset used for training
  subset      Subset to calculate the acc. on

optional arguments:
  -t T        Top-k error metric (default: 1)
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
parser.add_argument("subset", type=str, help="Subset to calculate the acc. on")
parser.add_argument("-t", type=int, help="Top-k error metric (default: 1)")
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
subset = args.subset

# Top-k error metric (default: 1)
k = args.t if args.t else 1  # Default: Top-1


# Loads the model
model = load_model(dataset_name, model_name)


# Loads the specified subset from the dataset.
images, labels = getattr(data_loader, subset)(dataset_name)


# Computes the Top-k acccuracy of the model.
def accuracy(images, labels, k):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    compteur = 0
    for (x, y) in tqdm(loader):
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        compteur += (y_pred.max(1)[1] == y).double().data.sum()
        # .double(): ByteTensor sums are limited at 256!
    return 100 * compteur / len(images)


"""
# Computes the precision@k for the specified values of k
def accuracy(images, labels, topk=(1,)):
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred = images.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
"""

# Prints the losses and accuracies at the end of each epoch.
acc = accuracy(images, labels, k)
print(f"acc: {acc:0.2f}%")
