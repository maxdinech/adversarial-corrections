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


# Creates the files `train.pt`, and `test.pt` from MNIST.
def create():
    dsets.MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor(),
                download=True)
    shutil.move('data/processed/test.pt', 'data/test.pt')
    shutil.move('data/processed/training.pt', 'data/train.pt')
    shutil.rmtree('data/raw')
    shutil.rmtree('data/processed')


# Loads a specified database.
def load(db_name, nb_elements):
    path = "data/" + db_name + ".pt"
    if not os.path.exists(path):
        create()
    images, labels = torch.load(path)
    images, labels = images[:nb_elements], labels[:nb_elements]
    if torch.cuda.is_available():
        images = images.type(torch.cuda.FloatTensor) / 255
        labels = labels.type(torch.cuda.LongTensor)
    else:
        images = images.type(torch.FloatTensor) / 255
        labels = labels.type(torch.LongTensor)
    images = images.view(len(images), 1, 28, 28)
    return images, labels


train = lambda nb_train=60000: load('train', nb_train)

test = lambda nb_test=10000: load('test', nb_test)
