"""
Trains a specified architucture to classify a specified dataset.
The networks are defined in architectures.py

---

usage: python3 train.py [-num NUM] [-val VAL] [-lr LR] [-e E] [-bs BS] [-t T]
                        [-S] model dataset

positional arguments:
  model       Network architecture (defined in architectures.py)
  dataset     Dataset used for training

optional arguments:
  -num NUM    Number of images used (default: all)
  -val VAL    Images proportion in val (default: 1/6)
  -lr LR      Learning rate (default: specified in model class)
  -e E        Num. of epochs (default: specified in model class)
  -bs BS      batch size (default: specified in model class)
  -t T        Top-k error metric (default: 1)
  -S, --save  Saves the trained model (default: False)
"""


import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from basics import to_Var, load_architecture
import data_loader
import plot


# Parameters parsing
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
                    help="Network architecture (defined in architectures.py)")
parser.add_argument("dataset", type=str,
                    help="Dataset used for training")
parser.add_argument("-num", type=float,
                    help="Number of images used (default: all)")
parser.add_argument("-val", type=float,
                    help="Images proportion in val (default: 1/6)")
parser.add_argument("-lr", type=float,
                    help="Learning rate (default: specified in model class)")
parser.add_argument("-e", type=int,
                    help="Num. of epochs (default: specified in model class)")
parser.add_argument("-bs", type=int,
                    help="batch size (default: specified in model class)")
parser.add_argument("-t", type=int,
                    help="Top-k error metric (default: 1)")
parser.add_argument("-S", "--save", action="store_true",
                    help="Saves the trained model (default: False)")
args = parser.parse_args()

model_name = args.model
dset_name = args.dataset
num_img = args.num
val_split = args.val if args.val else 1/6
save_model = args.save

# Model instanciation
model = load_architecture(model_name)


# Loads model hyperparameters (if not specified in args)
batch_size = args.bs if args.bs else model.batch_size
lr = args.lr if args.lr else model.lr
epochs = args.e if args.e else model.epochs


# Loads model functions
loss_fn = model.loss_fn
optimizer = model.optimizer


# Loads the train databases, and splits in into train and val.
train_images, train_labels = data_loader.train(dset_name, num_img, val_split)
val_images, val_labels = data_loader.val(dset_name, num_img, val_split)
num_train = len(train_images)
num_val = len(val_images)


# DataLoader of the train images
train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

num_batches = len(train_loader)


# Computes the Top-k acccuracy of the model.
# (computing the accuracy mini-batch after mini-batch avoids memory overload)
def accuracy(images, labels, k=1):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=100, shuffle=False)
    count = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        count += (y_pred.max(1)[1] == y).double().data.sum()
        # .double(): ByteTensor sums are limited at 256.
    return 100 * count / len(images)


# Computes the precision@k for the specified values of k
def accuracy(images, labels, topk=(1,)):
    batch_size = labels.size(0)

    _, pred = images.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)



# Computes the loss of the model.
# (computing the loss mini-batch after mini-batch avoids memory overload)
def big_loss(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=100, shuffle=False)
    count = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        count += len(x) * loss_fn(y_pred, y).data[0]
    return count / len(images)


# NETWORK TRAINING
# ----------------

# Prints the hyperparameters before the training.
print(f"Train on {num_train} samples, val on {num_val} samples.")
print(f"Epochs: {epochs}, batch size: {batch_size}")
optimizer_name = type(optimizer).__name__
print(f"Optimizer: {optimizer_name}, learning rate: {lr}")
num_parameters = sum(param.numel() for param in model.parameters())
print(f"Parameters: {num_parameters}")
print(f"Save model : {save_model}\n")


# Custom progress bar.
def bar(data, e):
    epoch = f"Epoch {e+1}/{epochs}"
    left = "{desc}: {percentage:3.0f}%"
    right = "{elapsed} - ETA:{remaining} - {rate_fmt}"
    bar_format = left + " |{bar}| " + right
    return tqdm(data, desc=epoch, ncols=100, unit='b', bar_format=bar_format)


train_accs, val_accs = [], []
train_losses, val_losses = [], []

try:
    # Main loop over each epoch
    for e in range(epochs):

        # Secondary loop over each mini-batch
        for (x, y) in bar(train_loader, e):

            # Computes the network output
            y_pred = model.train()(to_Var(x))
            loss = loss_fn(y_pred, to_Var(y))

            # Optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculates accuracy and loss on the train database.
        train_acc = accuracy(train_images, train_labels)
        train_loss = big_loss(train_images, train_labels)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Calculates accuracy and loss on the validation database.
        val_acc = accuracy(val_images, val_labels)
        val_loss = big_loss(val_images, val_labels)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        # Prints the losses and accs at the end of each epoch.
        print(f"  └─> train_loss: {train_loss:6.4f}",
              f"- train_acc: {train_acc:5.2f}%",
              f"  -   val_loss: {val_loss:6.4f}",
              f"- val_acc: {val_acc:5.2f}%")

except KeyboardInterrupt:
    pass

# Saves the network if stated.
if save_model:
    path = os.path.join("..", "models", dset_name, model_name + ".pt")
    torch.save(model, path)
    # Saves the accs history graph
    plot.train_history(train_accs, val_accs)
    plt.savefig(path + model_name + ".png", transparent=True)


def discriminator_performance(x, y):
    y, y_pred = to_Var(y), model.eval()(to_Var(x))
    faux_pos = ((y_pred.max(1)[1] != y) * (y_pred.max(1)[1] == 0))
    faux_pos = faux_pos.double().data.sum()
    faux_neg = ((y_pred.max(1)[1] != y) * (y_pred.max(1)[1] == 1))
    faux_neg = faux_neg.double().data.sum()
    total = (y_pred.max(1)[1] != y).double().data.sum()
    return (faux_pos, faux_neg, total)
