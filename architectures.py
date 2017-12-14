"""
Network architectures.
"""


import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Training hyperparameters
        self.lr = 2e-4
        self.epochs = 40
        self.batch_size = 32
        # Network definition
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 10),
            nn.Softmax()
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten  # Flatten
        x = self.classifier(x)
        return x


class MLP_d(nn.Module):
    def __init__(self):
        super(MLP_d, self).__init__()
        # Training hyperparameters
        self.lr = 2e-4
        self.epochs = 40
        self.batch_size = 32
        # Network definition
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(28 * 28, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(120, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(120, 10),
            nn.Softmax()
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten  # Flatten
        x = self.classifier(x)
        return x


class MLP_bn(nn.Module):
    def __init__(self):
        super(MLP_bn, self).__init__()
        # Training hyperparameters
        self.lr = 2e-4
        self.epochs = 40
        self.batch_size = 32
        # Network definition
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 120),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(120),
            nn.Linear(120, 120),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(120),
            nn.Linear(120, 10),
            nn.Softmax()
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten  # Flatten
        x = self.classifier(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Training hyperparameters
        self.lr = 2e-4
        self.epochs = 40
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 40, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(40 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 10),
            nn.Softmax()
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class CNN_d(nn.Module):
    def __init__(self):
        super(CNN_d, self).__init__()
        # Training hyperparameters
        self.lr = 2e-4
        self.epochs = 40
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 40, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(40 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(120, 10),
            nn.Softmax()
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class CNN_bn(nn.Module):
    def __init__(self):
        super(CNN_bn, self).__init__()
        # Training hyperparameters
        self.lr = 3e-5
        self.epochs = 40
        self.batch_size = 32
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 40, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(40)
        )
        self.classifier = nn.Sequential(
            nn.Linear(40 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(120),
            nn.Linear(120, 10),
            nn.Softmax()
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
        self.lr = 2e-5
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
            nn.Softmax()
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
        self.lr = 2e-5
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
            nn.Softmax()
        )
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
