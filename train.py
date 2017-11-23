"""
Entraînement de reseaux PyTorch sur MNIST.

Syntaxe : python -i train.py CNN

Les réseaux sont définis dans architectures.py
"""


import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mnist_loader
import architectures


# Paramètres passés
nom_modele = sys.argv[1]
enregistrement = ((sys.argv + ["False"])[2] == "True")

# Hyperparamètres
# ---------------
nb_train = 50000
nb_test = 10000


# Importation du modèle, déplacement sur GPU si possible
model = getattr(architectures, nom_modele)()
if torch.cuda.is_available():
    model = model.cuda()


# Import des paramètres du modèle
batch_size = model.batch_size
lr = model.lr
epochs = model.epochs

# Import des fonctions du modèle
loss_fn = model.loss_fn
optimizer = model.optimizer


# Fonction de création de variable sur GPU si possible, CPU sinon
def to_Var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Chargement des bases de données
train_images, train_labels = mnist_loader.train(nb_train)
test_images, test_labels = mnist_loader.test(nb_test)

# Création du DataLoader
train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

nb_batches = len(train_loader)


# Fonction de calcul de la précision du réseau
def accuracy(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    compteur = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        compteur += (y_pred.max(1)[1] == y).double().data.sum()
        # .double() parce que sinon on a un ByteTensor de sum() limitée à 256 !
    return 100 * compteur / len(images)


# Fonction de calcul de l'erreur sur beaucoup de données d'un coup
# (pour éviter les dépassements de capacité)
def big_loss(images, labels):
    data = TensorDataset(images, labels)
    loader = DataLoader(data, batch_size=1000, shuffle=False)
    compteur = 0
    for (x, y) in loader:
        y, y_pred = to_Var(y), model.eval()(to_Var(x))
        compteur += len(x) * loss_fn(y_pred, y).data[0]
        # .double() parce que sinon on a un ByteTensor de sum() limitée à 256 !
    return compteur / len(images)


# ENTRAINEMENT DU RÉSEAU
# ----------------------

# Affichage des HP
print("Train sur {} éléments, test sur {} éléments.".format(nb_train, nb_test))
print("Epochs: {}, Batch size: {}".format(epochs, batch_size))
optimizer_name = type(optimizer).__name__
print("Optimiseur: {}, lr: {}".format(optimizer_name, lr))
nb_parametres = sum(param.numel() for param in model.parameters())
print("Paramètres: {}".format(nb_parametres))
print("Enregistrement : {}\n".format(enregistrement))


# Barre de progression
def bar(data, e):
    epoch = "Epoch {}/{}".format(e+1, epochs)
    left = "{desc}: {percentage:3.0f}%"
    right = "{elapsed} - ETA:{remaining} - {rate_fmt}"
    bar_format = left + " |{bar}| " + right
    return tqdm(data, desc=epoch, ncols=100, unit='b', bar_format=bar_format)


# Boucle principale sur chaque epoch
for e in range(epochs):

    # Boucle secondaire sur chaque mini-batch
    for (x, y) in bar(train_loader, e):

        # Propagation dans le réseau et calcul de l'erreur
        y_pred = model.train()(to_Var(x))
        loss = loss_fn(y_pred, to_Var(y))

        # Ajustement des paramètres
        model.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 10 == 0:
        # Calcul de l'erreur totale et de la précision sur la base train.
        acc = accuracy(train_images, train_labels)
        loss = big_loss(train_images, train_labels)

        # Calcul de l'erreur totale et de la précision sur la base test.
        test_acc = accuracy(test_images, test_labels)
        test_loss = big_loss(test_images, test_labels)

        print("  └-> loss: {:6.4f} - acc: {:5.2f}%  ─  "
              .format(loss, acc), end='')
        print("test_loss: {:6.4f} - test_acc: {:5.2f}%"
              .format(test_loss, test_acc))


# Enregistrement du réseau
if enregistrement:
    torch.save(model, 'models/' + nom_modele + '.pt')


def ascii_print(image):
    image = image.view(28, 28)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999) % 5], end='')
        print('')


def prediction(n):
    img = test_images[n].view(1, 1, 28, 28)
    pred = model.eval()(img)
    print("prédiction :", pred.max(1)[1].data[0])
    ascii_print(img.data)


def prediction_img(img):
    pred = model.eval()(img)
    print("prédiction :", pred.max(0)[1].data[0])
    ascii_print(img.data)


def affichages():
    import random
    import time
    while True:
        print("\033[H\033[J")
        prediction(random.randrange(1000))
        time.sleep(0.7)
