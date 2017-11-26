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


# Loads the #img_id image from the train database.
def load_image(img_id):
    images, _ = mnist_loader.train(img_id+1)
    return to_Var(images[img_id].view(1, 1, 28, 28))


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
def attack_fixed_norm(img_id, N, p=2, lr=1e-3):
    image = load_image(img_id)
    digit = prediction(image).data[0]
    attacker = Perturbator_A(N, p, lr)
    for i in range(1000):
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        attacker.optimizer.step()
        # Prints results
        adv_image = attacker.forward(image)
        conf = confidence(adv_image, digit).data[0]
        N_ = (adv_image - image).norm(p).data[0]
        print("N = {:5.5f}  ->  {:4}  -  conf: {:0.4f}, norm_{}(r): {:0.4f}"
              .format(N, i, conf, p, N_), end='\r')
        if prediction(adv_image).data[0] != digit:
            break
    return (i < 999)


# Searches the minimal div value that fools the network
def minimal_attack_dichotomy(img_id, p=2, a=0, b=4, lr=0.005):
    while b-a >= 0.001:
        c = (a+b)/2
        print()
        success = attack_fixed_norm(img_id, c, p, lr)
        if success:
            b = c
        else:
            a = c
    return b


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
        if conf.data[0] >= 0.4:
            return conf
        else:
            return conf + 5 * (1-conf) * norm


def minimal_attack(img_id, p=2, lr=1e-3):
    norms, probs = [], []
    image = load_image(img_id)
    digit = prediction(image).data[0]
    attacker = Perturbator_B(p, lr)
    for i in range(500):
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        attacker.optimizer.step()
        # Prints results
        adv_image = attacker.forward(image)
        conf = confidence(adv_image, digit).data[0]
        N = (adv_image - image).norm(p).data[0]
        print("Step {:4} -- conf: {:0.4f}, norm_{}(r): {:0.10f}"
              .format(i, conf, p, N), end='\r')
    adv_image = attacker.forward(image)
    N = (adv_image - image).norm(p).data[0]
    success = prediction(adv_image).data[0] != digit
    return success, N


# Run multiple attacks
def minimal_attack_stats(nb):
    norms = []
    failed = 0
    for img_id in range(nb):
        success, N = minimal_attack(img_id)
        print()
        if success:
            norms.append(N)
        else:
            failed += 1
    # the histogram of the data
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['axes.titlepad'] = 10
    rcParams['font.family'] = "serif"
    rcParams['font.serif'] = "cm"
    rcParams['font.size'] = 8
    n, bins, patches = plt.hist(norms, 50, normed=1, facecolor='b', alpha=0.75)
    plt.xlabel("Norm")
    plt.ylabel("Probability")
    plt.title("Suceeded attacks : {}/{}".format(nb-failed, nb))
    plt.grid(True)
    plt.savefig("attack_results/" + model_name + ".png", transparent=True)
