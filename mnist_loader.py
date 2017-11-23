"""
Automatically creates and loads the MNIST database.

The database is split between train (50000 samples), test and val (10000
samples each).
"""


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import shutil
import os.path


# Creates the files `train.pt`, `test.pt` and `val.pt` from MNIST.
def create():
    dsets.MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor(),
                download=True)
    # On appelle `val` les images de `test`
    shutil.move('data/processed/test.pt', 'data/val.pt')
    # On divise `training` en `train` et `test`
    images, labels = torch.load('data/processed/training.pt')
    train_images, train_labels = images[:50000], labels[:50000]
    val_images, val_labels = images[50000:], labels[50000:]
    torch.save((train_images, train_labels), "data/train.pt")
    torch.save((val_images, val_labels), "data/test.pt")
    # On supprimme les dossiers temporaires
    shutil.rmtree('data/raw')
    shutil.rmtree('data/processed')


# Loads a specified database.
def load(db_name, nb_elements):
    url = "data/" + db_name + ".pt"
    if not os.path.exists(url):
        create()
    images, labels = torch.load(url)
    images, labels = images[:nb_elements], labels[:nb_elements]
    if torch.cuda.is_available():
        images = images.type(torch.cuda.FloatTensor) / 255
        labels = labels.type(torch.cuda.LongTensor)
    else:
        images = images.type(torch.FloatTensor) / 255
        labels = labels.type(torch.LongTensor)
    images = images.view(len(images), 1, 28, 28)
    return images, labels


train = lambda nb_train=50000: load('train', nb_train)

test = lambda nb_test=10000: load('test', nb_test)

val = lambda nb_val=10000: load('val', nb_val)
