"""
Adversarial examples creation against a specified model

Syntax: python -i attack.py CNN
"""


import sys
import torch
from torch import nn
from torch.autograd import Variable
import mnist_loader
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Passed parameters
model_name = sys.argv[1]


# Loads the database
images, labels = mnist_loader.test()


# Loads the #img_id image from the test database.
def load_image(img_id):
    return to_Var(images[img_id].view(1, 1, 28, 28))


# Loads the #img_id label from the test database.
def load_label(img_id):
    return to_Var(torch.Tensor([labels[img_id]]))


# Loads the model
try:
    model = torch.load("models/" + model_name + ".pt",
                       map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        model = model.cuda()
except FileNotFoundError:
    print("No model found")


# Returns a Variable containing `tensor`, on the GPU if CUDA is available.
def to_Var(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, requires_grad=requires_grad)


# Plots and saves the comparison graph
def compare(img_id, image, adv_image, p):
    model_name_tex = model_name.replace('_', '\\_')
    r = (adv_image - image)
    image_pred = prediction(image).data[0]
    image_prob = 100 * model.forward(image)[0, image_pred].data[0]
    adv_image_pred = prediction(adv_image).data[0]
    adv_image_prob = 100 * model.forward(adv_image)[0, adv_image_pred].data[0]
    norm = r.norm(p).data[0]
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['axes.titlepad'] = 10
    rcParams['font.family'] = "serif"
    rcParams['font.serif'] = "cm"
    rcParams['font.size'] = 8
    fig = plt.figure(figsize=(7, 2.5), dpi=180)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img)}} = {} \\small{{({:0.0f}\\%)}}"
              .format(model_name_tex, image_pred, image_prob))
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(r.data.view(28, 28).cpu().numpy(), cmap='RdBu')
    plt.title("Perturbation : $\\Vert r \\Vert_{{{}}} = {:0.4f}$"
              .format(p, round(norm, 3)))
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(adv_image.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img+r)}} = {} \\small{{({:0.0f}\\%)}}"
              .format(model_name_tex, adv_image_pred, adv_image_prob))
    plt.axis('off')
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.05)
    image_name = model_name + "_adv{}_n{}.png".format(img_id, p)
    plt.savefig("attack_results/" + image_name, transparent=True)
    plt.show()


# Plots an image (Variable)
def plot_image(image):
    plt.imshow(image.data.view(28, 28).numpy(), cmap='gray')
    plt.show()


# Returns the label prediction of an image.
def prediction(image):
    return model.eval()(image).max(1)[1]


# Returns the confidence of the network that the image is `digit`.
def confidence(image, digit):
    return model.eval()(image)[0, digit]


# METHOD A. - DICHOTOMY
# ---------------------

class Perturbator_A(nn.Module):
    """docstring for Perturbator_A"""
    def __init__(self, N, p, lr):
        super(Perturbator_A, self).__init__()
        self.N = N
        self.p = p
        self.r = nn.Parameter(torch.zeros(1, 1, 28, 28), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, image):
        r = self.r.clamp(-1, 1)
        return (image + (r * self.N / (1e-5 + r.norm(self.p)))).clamp(0, 1)
        # the 1-e5 added to the norm prevents r from diverging to nan

    def loss_fn(self, image, digit):
        adv_image = self.forward(image)
        return confidence(adv_image, digit)


# Network attack for a fixed norm_p value N.
def fixed_norm_attack(img_id, N, p=2, lr=1e-3):
    steps = 100
    image = load_image(img_id)
    digit = prediction(image).data[0]
    attacker = Perturbator_A(N, p, lr)
    for i in range(steps):
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        attacker.optimizer.step()
        # Prints results
        adv_image = attacker.forward(image)
        conf = confidence(adv_image, digit).data[0]
        N_ = (adv_image - image).norm(p).data[0]
        print("N = {:5.5f}  ->  step {:4}  -  conf: {:0.4f}, L_{}(r): {:0.4f}"
              .format(N, i, conf, p, N_), end='\r')
        if prediction(adv_image).data[0] != digit:
            break
    return (i < steps - 1), image, adv_image


# Searches the minimal div value that fools the network
def dichotomy_attack(img_id, p=2, a=0, b=8, lr=0.005):
    while b-a >= 0.001:
        c = (a+b)/2
        print()
        success, image, adv_image = fixed_norm_attack(img_id, c, p, lr)
        if success:
            b = c
            image_, adv_image_ = image, adv_image
        else:
            a = c
    print("\n\nMinimal N value found: ", b)
    compare(img_id, image_, adv_image_, p)


# METHOD B. - CUSTOM LOSS FUNCTION
# --------------------------------

class Perturbator_B(nn.Module):
    """docstring for Perturbator_B"""
    def __init__(self, p, lr):
        super(Perturbator_B, self).__init__()
        self.p = p
        self.r = nn.Parameter(torch.zeros(1, 1, 28, 28), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        r = self.r.clamp(-1, 1)
        return (x + r).clamp(0, 1)

    def loss_fn(self, image, digit):
        adv_image = self.forward(image)
        conf = model.eval()(adv_image)[0, digit]
        norm = (adv_image - image).norm(self.p)
        if (conf <= 0.2).data[0]:
            return norm
        elif (conf <= 0.3).data[0]:
            return conf + 5 * norm
        elif (conf <= 0.4).data[0]:
            return conf + norm
        elif (conf <= 0.95).data[0]:
            return 5 * conf + norm
        else:
            return 5 * conf - norm


def minimal_attack(image, p=2, lr=1e-3):
    steps = 100
    norms, confs = [], []
    digit = prediction(image).data[0]
    attacker = Perturbator_B(p, lr)
    if torch.cuda.is_available:
        attacker = attacker.cuda()
    optim = attacker.optimizer
    for i in range(steps):
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        optim.step()
        # Prints results
        adv_image = attacker.forward(image)
        conf = confidence(adv_image, digit).data[0]
        N = (adv_image - image).norm(p).data[0]
        print("Step {:4} -- conf: {:0.4f}, L_{}(r): {:0.10f}"
              .format(i, conf, p, N), end='\r')
        norms.append(N)
        confs.append(conf)
    adv_image = attacker.forward(image)
    success = prediction(adv_image).data[0] != digit
    return(success, adv_image, norms, confs)


def minimal_attack_graph(image_id, p=2, lr=1e-3):
    image = load_image(image_id)
    digit = prediction(image).data[0]
    success, adv_image, norms, confs = minimal_attack(image, p, lr)
    t = list(range(len(norms)))
    plt.plot(t, norms, 'r', t, confs, 'b')
    if prediction(adv_image).data[0] != digit:
        print("\nAttack suceeded")
        plt.show()
        compare(image_id, image, adv_image, p)
    else:
        print("\nAttack failed")

import random


def erreurs(n):
    i = 0
    l = len(images)
    while i < l and n > 0:
        image, label = load_image(i), load_label(i)
        if prediction(image).data[0] != label.data[0]:
            yield i
            n -= 1
        i += 1


def contre_attaque(image_id):
    image = load_image(image_id)
    adv_image = minimal_attack(image)[1]
    print()
    adv_adv_image = minimal_attack(adv_image)[1]
    l1 = prediction(image).data[0]
    l2 = prediction(adv_image).data[0]
    l3 = prediction(adv_adv_image).data[0]
    print("\n", l1, "->", l2, "->", l3, "\n")


def contre_attaques():
    while True:
        image = load_image(random.randint(0, 9999))
        adv_image = minimal_attack(image)[1]
        print()
        adv_adv_image = minimal_attack(adv_image)[1]
        print()
        adv_adv_adv_image = minimal_attack(adv_adv_image)[1]
        l1 = prediction(image).data[0]
        l2 = prediction(adv_image).data[0]
        l3 = prediction(adv_adv_image).data[0]
        l4 = prediction(adv_adv_adv_image).data[0]
        print("\n", l1, "->", l2, "->", l3, "->", l4, "\n")


# Run multiple attacks
def attacks():
    for img_id in range(10):
        for p in [2, 3, 5, 10]:
            minimal_attack(img_id, p)


def eureka(n):
    t = []
    for i in range(n):
        image, label = load_image(i), int(load_label(i).data[0])
        label_pred = int(prediction(image).data[0])
        est_erreur = label_pred != label
        _, adv_image, adv_norm, _ = minimal_attack(image)
        adv_norm = adv_norm[-1]
        adv_label = int(prediction(adv_image).data[0])
        err_rattrapee = adv_label == label
        t.append((est_erreur, err_rattrapee, adv_norm))
    return t


def gain(t, critere):
    nb_images = len(t)
    nb_erreurs = 0
    for i in t:
        est_erreur, err_rattrapee, adv_norm = i
        if est_erreur:  # Si c'était une erreur :
            if not err_rattrapee or adv_norm > critere:
                nb_erreurs += 1
        else:  # Si ce n'était pas une erreur :
            if adv_norm < critere:
                nb_erreurs += 1
    print("{}/{}".format(nb_erreurs, nb_images))
