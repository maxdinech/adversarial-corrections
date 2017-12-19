"""
Networks training.

Syntax : python train.py MODEL bool

If bool=True, the trained model is saved in models/

The models architectures are defined in architectures.py
"""


import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from basics import to_Var, load_architecture
import matplotlib.pyplot as plt
import plot
from tqdm import tqdm
import mnist_loader


# Passed parameters
model_name = sys.argv[1]
save_model = ((sys.argv + ["False"])[2] == "True")  # Default: save_model=False


# Sizes of the train and test databases
nb_train = 50000
nb_test = 10000


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
train_images, train_labels = mnist_loader.train(nb_train)
test_images, test_labels = mnist_loader.test(nb_test)

train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

nb_batches = len(train_loader)


# Computes the acccuracy of the model.
def accuracy(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    count = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        count += (y_pred.max(1)[1] == y).double().data.sum()
        # .double(): ByteTensor sums are limited at 256!
    return 100 * count / len(images)


# Computes the loss of the model.
# (computing the loss mini-match after mini-batch avoids memory overload)
def big_loss(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    count = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        count += len(x) * loss_fn(y_pred, y).data[0]
        # .double() to avoid being limited at 256 (ByteTensor) !
    return count / len(images)


# NETWORK TRAINING
# ----------------

# Prints the hyperparameters before the training.
print("Train on {} samples, test on {} samples.".format(nb_train, nb_test))
print("Epochs: {}, batch size: {}".format(epochs, batch_size))
optimizer_name = type(optimizer).__name__
print("Optimizer: {}, learning rate: {}".format(optimizer_name, lr))
nb_parameters = sum(param.numel() for param in model.parameters())
print("Parameters: {}".format(nb_parameters))
print("Save model : {}\n".format(save_model))


# Custom progress bar.
def bar(data, e):
    epoch = "Epoch {}/{}".format(e+1, epochs)
    left = "{desc}: {percentage:3.0f}%"
    right = "{elapsed} - ETA:{remaining} - {rate_fmt}"
    bar_format = left + " |{bar}| " + right
    return tqdm(data, desc=epoch, ncols=100, unit='b', bar_format=bar_format)


# Main loop over each epoch
train_accs, test_accs = [], []
train_losses, test_losses = [], []
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

    # Calculates accuracy and loss on the test database.
    test_acc = accuracy(test_images, test_labels)
    test_loss = big_loss(test_images, test_labels)
    test_accs.append(test_acc)
    test_losses.append(test_loss)

    # Prints the losses and accs at the end of each epoch.
    print("  └-> train_loss: {:6.4f} - train_acc: {:5.2f}%  ─  "
          .format(train_loss, train_acc), end='')
    print("test_loss: {:6.4f} - test_acc: {:5.2f}%"
          .format(test_loss, test_acc))


# Saves the network if stated.
if save_model:
    file = open('models/results.txt', 'a')
    file.write("{}: train_acc: {}  -  test_acc: {}"
               .format(model_name, train_acc, test_acc))
    torch.save(model, 'models/' + model_name + '.pt')
    # Saves the accs history graph
    plot.accs(train_accs, test_accs)
    plt.savefig("models/" + model_name + ".png", transparent=True)
