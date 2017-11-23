"""
GAN simple avec PyTorch sur MNIST.

TODO:
    - Expérimenter avec les CNN et les déconvolutions.

"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import mnist_loader


# Hyperparamètres
# ---------------
epochs = 200
batch_size = 10
G_lr = 0.0003
D_lr = 0.0003
chiffre = 3


# Création de variable sur GPU si possible, CPU sinon
def to_Var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Chargement de la base de données
images, labels = mnist_loader.train(60_000, flatten=True)
indices = [i for i in range(len(images)) if labels[i] == chiffre]
images = images[indices]

# Création du DataLoader
data_loader = DataLoader(images,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

# Conversion de la BDD en Variable
images = to_Var(images)


# Réseau Discriminateur
D = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid())


# Réseau Générateur
G = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Sigmoid())


# Déplacement vers le GPU si possible
if torch.cuda.is_available():
    D.cuda()
    G.cuda()


# Fonction d'erreur
loss_fn = nn.BCELoss()


# Optimiseur : Adam
D_optimizer = torch.optim.Adam(D.parameters(), lr=D_lr)
G_optimizer = torch.optim.Adam(G.parameters(), lr=G_lr)


# Générateur de bruit aléatoire (z)
def random_noise(n):
    return to_Var(torch.randn(n, 64))


# Entraînement de D et G
for epoch in range(epochs):
    print("Epoch {}/{}:".format(epoch + 1, epochs))

    for i, images in enumerate(data_loader):
        
        indice = str((i+1)).zfill(3)
        print("└─ ({}/{}) ".format(indice, len(data_loader)), end='')
        p = int(20 * i / len(data_loader))
        print('▰'*p + '▱'*(20-p), end='    ')

        # Labels des vraies et fausses entrées
        real_labels = to_Var(torch.ones(batch_size, 1))
        fake_labels = to_Var(torch.zeros(batch_size, 1))

        #=============== entraînement de D ===============#
        real_images = to_Var(images.view(batch_size, -1))
        D_pred_real = D(real_images)
        D_loss_real = loss_fn(D_pred_real, real_labels)

        fake_images = G(random_noise(batch_size))
        D_pred_fake = D(fake_images)
        D_loss_fake = loss_fn(D_pred_fake, fake_labels)
        
        D_loss = D_loss_real + D_loss_fake

        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        #=============== entraînement de G ===============#

        fake_images = G(random_noise(batch_size))
        D_pred_fake = D(fake_images)
        G_loss = loss_fn(D_pred_fake, real_labels)

        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()


        print('D_loss: %.4f,   G_loss: %.4f,   D(x): %.2f,   D(G(z)): %.2f' 
              %(D_loss.data[0], G_loss.data[0],
                D_pred_real.data.mean(), D_pred_fake.data.mean()), end='\r')

    # Calcul et affichage de loss et acc à chaque fin d'epoch
    
    img1 = G.forward(random_noise(1)).data
    img2 = G.forward(random_noise(1)).data
    ascii_print_sided(img1, img2)


while True:
    print("\033[H\033[J")
    img1 = G.forward(random_noise(1)).data
    img2 = G.forward(random_noise(1)).data
    ascii_print_sided(img1, img2)
    time.sleep(0.7)