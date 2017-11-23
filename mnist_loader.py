"""
Crée automatiquement si besoin et charge la base de données MNIST

La base de données est découpée en train, test et val (voir README.md)
"""


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os.path
import shutil


# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


# Création des fichiers train.pt, test.pt et val.pt de MNIST
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


def load(nom, nb_elements):
    url = "data/" + nom + ".pt"
    if not os.path.exists(url):
        create()
    images, labels = torch.load(url)
    images, labels = images[:nb_elements], labels[:nb_elements]
    images = images.type(dtype) / 255
    images = images.view(len(images), 1, 28, 28)
    return images, labels


train = lambda nb_train=50000: load('train', nb_train)

test = lambda nb_test=10000: load('test', nb_test)

val = lambda nb_val=10000: load('val', nb_val)
