"""
Networks training.

Syntax : python -i train.py CNN False

The networks are defined in architectures.py
"""


import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import mnist_loader
import architectures


# Passed parameters
model_name = sys.argv[1]
save_model = ((sys.argv + ["False"])[2] == "True")  # Default: save_model=False


# Hyperparameters
nb_train = 50000
nb_test = 10000


# Model instanciation
model = getattr(architectures, model_name)()
if torch.cuda.is_available():
    model = model.cuda()


# Loads model hyperparameters
batch_size = model.batch_size
lr = model.lr
epochs = model.epochs

# Loads model functions
loss_fn = model.loss_fn
optimizer = model.optimizer


# Returns a Variable containing `tensor`, on the GPU if CUDA is available.
def to_Var(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, requires_grad=requires_grad)


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
    compteur = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        compteur += (y_pred.max(1)[1] == y).double().data.sum()
        # .double() parce que sinon on a un ByteTensor de sum() limitée à 256 !
    return 100 * compteur / len(images)


# Computes the loss of the model.
# (computing the loss mini-match after mini-batch avoids memory overload)
def big_loss(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    compteur = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        compteur += len(x) * loss_fn(y_pred, y).data[0]
        # .double() to avoid being limited at 256 (ByteTensor) !
    return compteur / len(images)


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
accuracies, test_accuracies = [], []
losses, test_losses = [], []
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
    acc = accuracy(train_images, train_labels)
    loss = big_loss(train_images, train_labels)
    accuracies.append(acc)
    losses.append(loss)

    # Calculates accuracy and loss on the test database.
    test_acc = accuracy(test_images, test_labels)
    test_loss = big_loss(test_images, test_labels)
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)

    # Prints the losses and accuracies at the end of each epoch.
    print("  └-> loss: {:6.4f} - acc: {:5.2f}%  ─  "
          .format(loss, acc), end='')
    print("test_loss: {:6.4f} - test_acc: {:5.2f}%"
          .format(test_loss, test_acc))


# Saves the network if stated.
if save_model:
    torch.save(model, 'models/' + model_name + '.pt')
    # Saves the accuracies history graph
    t = list(range(epochs))
    plt.plot(t, accuracies, 'r')
    plt.plot(t, test_accuracies, 'b')
    plt.title("Network training history")
    plt.legend(["acc", "test_acc"])
    plt.savefig("models/" + model_name + "_acc.png", transparent=True)
    # Saves the losses history graph
    plt.clf()
    plt.plot(t, losses, 'r')
    plt.plot(t, test_losses, 'b')
    plt.title("Network training history")
    plt.legend(["loss", "test_loss"])
    plt.savefig("models/" + model_name + "_loss.png", transparent=True)
