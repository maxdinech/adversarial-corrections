"""
Networks training.

Syntax : python test.py CNN

The networks are defined in architectures.py
"""


import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from basics import to_Var
from tqdm import tqdm
import fashion_mnist_loader


# Passed parameters
model_name = sys.argv[1]


# Loads the model
try:
    model = torch.load("models/" + model_name + ".pt",
                       map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        model = model.cuda()
except FileNotFoundError:
    print("No model found")


# Loads the train, test and val databases.
train_images, train_labels = fashion_mnist_loader.train()
test_images, test_labels = fashion_mnist_loader.test()


# Computes the acccuracy of the model.
def accuracy(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    compteur = 0
    for (x, y) in tqdm(loader):
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        compteur += (y_pred.max(1)[1] == y).double().data.sum()
        # .double(): ByteTensor sums are limited at 256!
    return 100 * compteur / len(images)


train_acc = accuracy(train_images, train_labels)
test_acc = accuracy(test_images, test_labels)


# Prints the losses and accuracies at the end of each epoch.
print("train_acc: {:0.2f}%  ─  test_acc: {:0.2f}%"
      .format(train_acc, test_acc))
