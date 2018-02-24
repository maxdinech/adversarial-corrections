"""
Network architectures.
"""


import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Training hyperparameters
        self.lr = 2e-4
        self.epochs = 40
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(1, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(5, 10, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1)
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(len(x), 1, 1, 50)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Training hyperparameters
        self.lr = 5e-4
        self.epochs = 100
        self.batch_size = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(120, 10),
            nn.Softmax(dim=1)
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class AlexNet_bn(nn.Module):
    def __init__(self):
        super(AlexNet_bn, self).__init__()
        # Training hyperparameters
        self.lr = 5e-4
        self.epochs = 100
        self.batch_size = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(120),
            nn.Linear(120, 10),
            nn.Softmax(dim=1)
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Training hyperparameters
        self.lr = 1e-4
        self.epochs = 40
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class VGG_bn(nn.Module):
    def __init__(self):
        super(VGG_bn, self).__init__()
        # Training hyperparameters
        self.lr = 1e-4
        self.epochs = 40
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
