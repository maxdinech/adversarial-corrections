"""
Automatically creates and loads the MNIST and FashionMNIST datasets.
"""


import os
import shutil

import torch
from torchvision import datasets, transforms


# Creates `train.pt` and `test.pt` from the specified dataset.
def create(dataset):
    root = os.path.join('..', 'data')
    if dataset == 'MNIST':
        datasets.MNIST(root=root, train=True,
                       transform=transforms.ToTensor(),
                       download=True)
    elif dataset == 'FashionMNIST':
        datasets.FashionMNIST(root=root, train=True,
                              transform=transforms.ToTensor(),
                              download=True)
    os.mkdir(os.path.join(root, dataset))
    shutil.move(os.path.join(root, 'processed', 'training.pt'),
                os.path.join(root, dataset, 'train.pt'))
    shutil.move(os.path.join(root, 'processed', 'test.pt'),
                os.path.join(root, dataset, 'test.pt'))
    shutil.rmtree(os.path.join(root, 'raw'))
    shutil.rmtree(os.path.join(root, 'processed'))


# Loads a subset from a dataset.
def load(dataset, subset, num_elements=None):
    root = os.path.join('..', 'data')
    if dataset in ['MNISTnorms', 'FashionMNISTnorms',
                   'MNISTconfs', 'FashionMNISTconfs']:
        file_name = subset + '_' + dataset[-5:] + '.pt'
        path = os.path.join(root, dataset[:-5], file_name)
        values, labels = torch.load(path)
        return values, labels.long()
    elif dataset in ['MNIST', 'FashionMNIST']:
        path = os.path.join(root, dataset, subset + '.pt')
        if not os.path.exists(path):
            create(dataset)
        images, labels = torch.load(path)
        if num_elements:
            images = images[:num_elements].clone()
            labels = labels[:num_elements].clone()
        images = images.float() / 255
        labels = labels.long()
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(len(images), 1, 28, 28)  # Channels first
        return images, labels
    else:
        raise ValueError('Unknown dataset')


# Loads ((1 - val_split) * num_images) from `train` starting at position 0
def train(dataset, val_split, num_images=None):
    images, labels = load(dataset, 'train', num_images)
    num_images = num_images if num_images else len(images)
    num_train = round((1-val_split) * num_images)
    train_images = images[:num_train].clone()
    train_labels = labels[:num_train].clone()
    return train_images, train_labels


# Loads (val_split * num_images) from `train` starting at position `num_train`
def val(dataset, val_split, num_images=None):
    images, labels = load(dataset, 'train', num_images)
    num_images = num_images if num_images else len(images)
    num_train = round((1-val_split) * num_images)
    train_images = images[num_train:num_images].clone()
    train_labels = labels[num_train:num_images].clone()
    return train_images, train_labels


# Loads the test images
# The unused second argument gives the same type to the three functions.
def test(dataset, _, num_test=None):
    return load(dataset, 'test', num_test)
