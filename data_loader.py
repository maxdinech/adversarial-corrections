"""
Automatically creates and loads the MNIST and FashionMNIST datasets.

The datasets are split between :
    - train (50000 samples),
    - test  (10000 samples),
    - val   (10000 samples).
"""


import torch
from torchvision import datasets, transforms
import shutil
import os.path


# Creates the files `train.pt`, and `test.pt` from FashionMNIST.
def create(dataset):
    if dataset == 'MNIST':
        datasets.MNIST(root='data/',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)
    elif dataset == 'FashionMNIST':
        datasets.FashionMNIST(root='data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)
    else:
        raise ValueError("Unknown dataset")
    os.mkdir('data/' + dataset)
    images, labels = torch.load('data/processed/training.pt')
    images_train, labels_train = images[:50000].clone(), labels[:50000].clone()
    images_val, labels_val = images[50000:].clone(), labels[50000:].clone()
    torch.save((images_train, labels_train), 'data/' + dataset + '/train.pt')
    torch.save((images_val, labels_val), 'data/' + dataset + '/val.pt')
    shutil.move('data/processed/test.pt', 'data/' + dataset + '/test.pt')
    shutil.rmtree('data/raw')
    shutil.rmtree('data/processed')


# Loads a specified database.
def load(dataset, db_name, nb_elements):
    if dataset not in ['MNIST', 'FashionMNIST']:
        raise ValueError("Unknown dataset")
    else:
        path = 'data/' + dataset + '/' + db_name + '.pt'
        if not os.path.exists(path):
            create(dataset)
        images, labels = torch.load(path)
        images = images[:nb_elements].clone()
        labels = labels[:nb_elements].clone()
        images = images.float() / 255
        labels = labels.long()
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(len(images), 1, 28, 28)
    return images, labels


train = lambda dataset, nb_train=50000: load(dataset, 'train', nb_train)

test = lambda dataset, nb_test=10000: load(dataset, 'test', nb_test)

val = lambda dataset, nb_val=10000: load(dataset, 'val', nb_val)
