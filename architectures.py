"""
Architectures de réseaux PyTorch utilisés pour la reconaissance de MNIST.
Ce fichier permet de partager les architectures utilisées avec adversarial.py

Chaque réseau contient ses informations d'apprentissage :
    - lr, epochs et batch_size
    - loss_fn
    - optimizer

Ce qui permet de ne pas avoir à changer tous les HP à chaque fois que l'on
change de réseau.

"""


import torch
from torch import nn
import torch.nn.functional as F


# MLP à deux couches cachées (28*28 -> 128 -> 128 -> 10)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Hyperparamètres d'entrainement
        self.lr = 2e-4
        self.epochs = 30
        self.batch_size = 32
        # Définition des couches du modèle
        self.fc1 = nn.Linear(28*28, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)
        # fonctions d'erreur et optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


# MLP à deux couches cachées (avec dropout) (28*28 -> 128 -> 128 -> 10)
class MLP_d(nn.Module):
    def __init__(self):
        super(MLP_d, self).__init__()
        # Hyperparamètres d'entrainement
        self.lr = 2e-4
        self.epochs = 30
        self.batch_size = 32
        # Définition des couches du modèle
        self.fc1 = nn.Linear(28*28, 120)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(120, 120)
        self.drop2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(120, 10)
        # fonctions d'erreur et optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.softmax(self.fc3(x))
        return x


# CNN à deux convolutions
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Hyperparamètres d'entrainement
        self.lr = 2e-4
        self.epochs = 30
        self.batch_size = 32
        # Définition des couches du modèle
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5*5*40, 120)
        self.fc2 = nn.Linear(120, 10)
        # fonctions d'erreur et optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


# CNN à deux convolutions (avec dropout)
class CNN_d(nn.Module):
    def __init__(self):
        super(CNN_d, self).__init__()
        # Hyperparamètres d'entrainement
        self.lr = 2e-4
        self.epochs = 30
        self.batch_size = 32
        # Définition des couches du modèle
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5*5*40, 120)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(120, 10)
        self.drop2 = nn.Dropout(p=0.4)
        # fonctions d'erreur et optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)  # Flatten
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.softmax(self.fc2(x))
        return x
