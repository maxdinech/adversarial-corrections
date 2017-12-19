"""
Adversarial examples creation against a specified model

Syntax: python -i attack.py MODEL
"""


import sys
import torch
from torch import nn
from basics import to_Var, load_model
import mnist_loader
import matplotlib.pyplot as plt
import plot

# Passed parameters
model_name = sys.argv[1]


# Loads the model
model = load_model(model_name)


# Loads the database
images, labels = mnist_loader.test()


# BASIC FUNCTIONS
# ---------------

# Loads the #img_id image from the test database.
def load_image(img_id):
    return to_Var(images[img_id].view(1, 1, 28, 28))


# Loads the #img_id label from the test database.
def load_label(img_id):
    return labels[img_id]


# Returns the label prediction of an image.
def prediction(image):
    return model.eval()(image).max(1)[1].data[0]


# Returns the confidence of the network that the image is `digit`.
def confidence(image, category):
    return model.eval()(image)[0, category].data[0]


def errors(n=len(images)):
    i = 0
    l = len(images)
    while i < l and n > 0:
        image, label = load_image(i), load_label(i)
        if prediction(image) != label:
            yield i
            n -= 1
        i += 1


# ATTACK FUNCTIONS
# ----------------

class Attacker(nn.Module):
    def __init__(self, p, lr):
        super(Attacker, self).__init__()
        self.p = p
        self.r = nn.Parameter(torch.zeros(1, 1, 28, 28), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return (x + self.r).clamp(0, 1)

    def loss_fn(self, image, digit):
        adv_image = self.forward(image)
        conf = model.eval()(adv_image)[0, digit]
        norm = (adv_image - image).pow(self.p).sum()
        if (conf < 0.2).data[0]:
            return norm
        elif (conf < 0.95).data[0]:
            return conf + norm
        else:
            return conf - norm


def attack(image, steps=500, p=2, lr=1e-3):
    norms, confs = [], []
    digit = prediction(image)
    attacker = Attacker(p, lr)
    if torch.cuda.is_available():
        attacker = attacker.cuda()
    optim = attacker.optimizer
    for i in range(steps):
        # Training step
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        optim.step()
        # Prints results
        adv_image = attacker.forward(image)
        conf = confidence(adv_image, digit)
        norm = (adv_image - image).norm(p).data[0]
        print("Step {:4} -- conf: {:0.4f}, L_{}(r): {:0.10f}"
              .format(i, conf, p, norm), end='\r')
        norms.append(norm)
        confs.append(conf)
    print()
    success = prediction(adv_image) != digit
    return(success, adv_image, norms, confs)


def attack_break(image, max_steps=500, p=2, lr=1e-3):
    norms, confs = [], []
    digit = prediction(image)
    attacker = Attacker(p, lr)
    if torch.cuda.is_available():
        attacker = attacker.cuda()
    optim = attacker.optimizer
    adv_image = attacker.forward(image)
    steps = 0
    while confidence(adv_image, digit) >= 0.2 and steps < max_steps:
        steps += 1
        # Training step
        loss = attacker.loss_fn(image, digit)
        attacker.zero_grad()
        loss.backward()
        optim.step()
        # Prints results
        adv_image = attacker.forward(image)
        conf = confidence(adv_image, digit)
        norm = (adv_image - image).norm(p).data[0]
        print("Step {:4} -- conf: {:0.4f}, L_{}(r): {:0.10f}"
              .format(steps, conf, p, norm), end='\r')
        norms.append(norm)
        confs.append(conf)
    print()
    return(steps, adv_image, norms, confs)


def attack_graph(image_id, steps=500, p=2, lr=1e-3):
    image = load_image(image_id)
    success, adv_image, norms, confs = attack(image, steps, p, lr)
    plot.attack_history(norms, confs)
    plt.show()
    if success:
        print("\nAttack suceeded")
        image_pred = prediction(image)
        image_conf = confidence(image, image_pred)
        adv_image_pred = prediction(adv_image)
        adv_image_conf = confidence(adv_image, adv_image_pred)
        plt.compare(model_name, image_id, p,
                    image, image_pred, image_conf,
                    adv_image, adv_image_pred, adv_image_conf)
        image_name = model_name + "_adv{}_n{}.png".format(image_id, p)
        plt.savefig("attack_results/" + image_name, transparent=True)
        plt.show()
    else:
        print("\nAttack failed")


def attack_break_graph(image_id, max_steps=500, p=2, lr=1e-3):
    image = load_image(image_id)
    success, adv_image, norms, confs = attack_break(image, max_steps, p, lr)
    plot.attack_history(norms, confs)
    plt.show()
    image_pred = prediction(image)
    image_conf = confidence(image, image_pred)
    adv_image_pred = prediction(adv_image)
    adv_image_conf = confidence(adv_image, adv_image_pred)
    plt.compare(model_name, image_id, p,
                image, image_pred, image_conf,
                adv_image, adv_image_pred, adv_image_conf)
    plt.show()


def multiple_attacks_graph(list, steps=500, p=2, lr=1e-3):
    plt.clf()
    for image_id in list:
        image = load_image(image_id)
        success, adv_image, norms, confs = attack(image, steps, p, lr)
        plot.attack_history(norms, confs)
    plt.savefig("/attack_results/multiple.png", transparent=True)
    plt.show()


# RESISTANCE FONCTIONS
# --------------------

def resistance_N(image_id, steps):
    norms = attack(load_image(image_id), steps)[2]
    return norms[-1]


def resistance_max(image_id, steps):
    norms = attack(load_image(image_id), steps)[2]
    return max(norms)


# Returns both the max-resistance and the N-resistance
def resistances(image_id, steps):
    norms = attack(load_image(image_id), steps)[2]
    return (norms[-1], max(norms))


def resistance_min(image_id, max_steps):
    return attack_break(load_image(image_id), max_steps)[0]


# STATS FUNCTIONS
# ---------------


def histogram(values, delimiters):
    tensor = torch.Tensor(values)
    counts = []
    for i in range(len(delimiters) - 1):
        inf = delimiters[i]
        sup = delimiters[i+1]
        counts += [((inf <= tensor) & (tensor < sup)).double().sum()]
    inf = delimiters[-1]
    counts += [(inf <= tensor).double().sum()]
    return counts


def resistances_lists(list, steps):
    L_res_N, L_res_max = [], []
    for image_id in list:
        res_N, res_max = resistances(image_id, steps)
        L_res_N += [res_N]
        L_res_max += [res_max]
    return (L_res_N, L_res_max)


# AVERSARIAL COUNTER-ATTACKS
# --------------------------

# Tests if the label obtained by the adversarial counter-attack is
# the true one.
def counter_attack(image_id, max_steps):
    image = load_image(image_id)
    label = load_label(image_id)
    adv_image = attack_break(image, max_steps)[1]
    adv_label = prediction(adv_image)
    return adv_label == label


def counter_attacks(list, max_steps):
    return [counter_attack(i, max_steps) for i in list]
