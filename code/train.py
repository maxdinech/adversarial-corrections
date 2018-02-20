"""
Networks training.

Syntax : python train.py MODEL DATASET SAVE

MODEL is either AlexNet, AlexNet_bn, VGG or VGG_bn (see architectures.py)

DATASET is either :
    - MNIST
    - FashionMNIST
    - MNISTnorms
    - MNISTconfs
    - FashionMNISTnorms
    - FashionMNISTconfs

If SAVE is True, the trained model is saved in models/DATASET/.

"""


import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from basics import to_Var, load_architecture
import matplotlib.pyplot as plt
import plot
from tqdm import tqdm
import data_loader


# Passed parameters
model_name = sys.argv[1]
dataset_name = sys.argv[2]
save_model = ((sys.argv + ["False"])[3] == "True")  # Default: save_model=False


# Sizes of the train and validation databases
nb_train = 50000
nb_val = 10000


# Model instanciation
model = load_architecture(model_name)


# Loads model hyperparameters
batch_size = model.batch_size
lr = model.lr
epochs = model.epochs

# Loads model functions
loss_fn = model.loss_fn
optimizer = model.optimizer


# Loads the train and test databases.
train_images, train_labels = data_loader.train(dataset_name, nb_train)
val_images, val_labels = data_loader.val(dataset_name, nb_val)

train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

nb_batches = len(train_loader)


# Computes the acccuracy of the model.
# (computing the accuracy mini-batch after mini-batch avoids memory overload)
def accuracy(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=100, shuffle=False)
    count = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        count += (y_pred.max(1)[1] == y).double().data.sum()
        # .double(): ByteTensor sums are limited at 256.
    return 100 * count / len(images)


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
print(f"Train on {nb_train} samples, val on {nb_val} samples.")
print(f"Epochs: {epochs}, batch size: {batch_size}")
optimizer_name = type(optimizer).__name__
print(f"Optimizer: {optimizer_name}, learning rate: {lr}")
nb_parameters = sum(param.numel() for param in model.parameters())
print(f"Parameters: {nb_parameters}")
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
    path = 'models/' + dataset_name + '/'
    torch.save(model, path + model_name + '.pt')
    # Saves the accs history graph
    plot.train_history(train_accs, val_accs)
    plt.savefig(path + model_name + ".png", transparent=True)


def performance(x, y):
    y, y_pred = to_Var(y), model.eval()(to_Var(x))
    faux_pos = ((y_pred.max(1)[1] != y) * (y_pred.max(1)[1] == 0))
    faux_pos = faux_pos.double().data.sum()
    faux_neg = ((y_pred.max(1)[1] != y) * (y_pred.max(1)[1] == 1))
    faux_neg = faux_neg.double().data.sum()
    total = (y_pred.max(1)[1] != y).double().data.sum()
    return (faux_pos, faux_neg, total)
