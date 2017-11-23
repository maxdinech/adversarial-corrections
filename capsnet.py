"""
Capsule Network avec PyTorch


La création d'une image se fait à partir de la superposition de figures qui
sont tournées, redimensionnées, et placées à des positions données du plan. Un
CapsNet est un réseau de neurones qui essaie de réaliser l'opération inverse.

On associe à chaque pixel de l'image la réponse de chaque capsule, ce qui donne
un champ de vecteurs. Leur norme représente leur probabilité, et leur
orientation un paramètre (taille, orientation, épaisseur...)

Par exemple : Une capsule peut servir à detecter les rectangles, elle
s'activera fortement aux endroits o* un rectangle est présent, et faiblement
sinon. L'orientation du vecteur peut par exemple représenter l'orientation du
triangle.

Distribution de probabilités, donc on veut que toutes les normes soient entre
zéro et un. On utilise pour celà la fonction de Squash, qui préserve
l'orientation :

                      ||u||²     u
        Squash(u) =  ------- * -----    (1)
                     1+||u||²  ||u||

On cherche ensuite à transférer l'information aux capsules de la couche
suivante. Pour ne pas bêtement transférer la même information à tout le monde,
on effectue un routage. Pour cela, les capsules vont chercher à prédire la
sortie de la capsule suivante en fonction de leur propre information.

Par exemple, si les trois capsules de premier niveau qui représentent les trois
éléments d'une maison ont la même orientation, taille et épaisseur, on peut
déjà prédire que cette information est pertitente : on la propage à la capsule
de deuxième niveau qui identifie les maisons.


"""

import torch
import torch.nn.functional as F
from torch import nn


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        # Hyperparamètres d'entrainement
        self.lr = 2e-4
        self.epochs = 30
        self.batch_size = 32
        # Définition des couches du modèle
        self.conv1 = nn.Conv2d(1, 256, 9)
        self.conv2 = nn.Conv2d(1, 256, 9, 2)
        # fonctions d'erreur et optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = x.view(len(x), 20, 20, 32, 8)  # 32 caps. primaires de dimension 8
        x = PrimaryCaps(x)
        x = DigiCaps(x)

        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

