"""
Génération d'examples adversaires sur PyTorch

"""


import sys
import torch
from torch.autograd import Variable
import mnist_loader
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Paramètres passés
nom_modele = sys.argv[1]


# Création de variable sur GPU si possible, CPU sinon
def to_Var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Importation du modèle
try:
    model = torch.load("models/" + nom_modele + ".pt",
                       map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        model = model.cuda()
except FileNotFoundError:
    print("Pas de modèle trouvé !")


def compare(image1, r, image2, num, p, norme):
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['axes.titlepad'] = 10
    rcParams['font.family'] = "serif"
    rcParams['font.serif'] = "cm"
    fig = plt.figure(figsize=(7, 2.5), dpi=200)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image1.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img)}} = {}"
              .format(nom_modele, prediction(image1)))
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(r.data.view(28, 28).cpu().numpy(), cmap='RdBu')
    plt.title("Perturbation : $\\Vert r \\Vert_{{{}}} = {}$"
              .format(p, round(norme, 3)))
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image2.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img + r)}} = {}"
              .format(nom_modele, prediction(image2)))
    plt.axis('off')
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.05)
    nom_image = "resultats/" + nom_modele + "_adv{}_n{}.png".format(num, p)
    plt.savefig(nom_image, transparent=True)
    # plt.show()


def affiche(image):
    plt.imshow(image.clamp(0, 1).data.view(28, 28).numpy(), cmap='gray')
    plt.show()

# Sélection d'une image dans la base de données (la première : un chiffre 5)


# Première méthode d'attaque : exploration du voisinage de l'image en cherchant
# à avoir une grande erreur tout en restant assez près On cherche la direction
# la plus favorable par calcul du gradient
#   1. On calcule et trie les gradients
#   2. On modifie l'image
# Jusqu'à obtenir une prédiction incorrecte


def charge_image(num):
    images, _ = mnist_loader.train(num+1)
    return to_Var(images[num].view(1, 1, 28, 28))


def prediction(image):
    return model.forward(image.clamp(0, 1)).max(1)[1].data[0]


def attaque(num, lr=0.005, div=0.2, p=2):
    image = charge_image(num)
    chiffre = prediction(image)
    r = to_Var(torch.zeros(1, 1, 28, 28), requires_grad=True)
    adv = lambda image, r: (image + (r * div / (1e-5 + r.norm(p)))).clamp(0, 1)
    image_adv = adv(image, r)
    i = 0
    while prediction(image_adv) == chiffre:
        loss = model.forward(image_adv)[0, chiffre]
        loss.backward()
        print(str(i).zfill(4), loss.data[0], end='\r')
        r.data -= lr * r.grad.data / r.grad.data.abs().max()
        r.grad.data.zero_()
        image_adv = adv(image, r)
        i += 1
        if i >= 1000:
            break
    return (i < 1000), image, (image_adv-image), image_adv


def attaque_optimale(num, p=2, a=0, b=4, lr=0.005):
    while b-a >= 0.001:
        c = (a+b)/2
        print("\n\nImage : {}, norme_{}(r) = {}\n".format(num, p, c))
        succes, img, r, adv = attaque(num, lr, c, p)
        if succes:
            b = c
        else:
            a = c
    print("\n\nValeur minimale approchée : ", b)
    compare(img, r, adv, num, p, b)


def attaques():
    for num in range(10):
        for p in [2, 3, 5, 10]:
            attaque_optimale(num, p, 0, 5)
