"""
Adversarial attacks

---

usage: python3 -i attack.py model dataset subset

positional arguments:
  model       Trained model to evaluate
  dataset     Dataset used for training
  subset      Subset to calculate the acc. on
"""


import os
import sys
import argparse

import torch
from torch import nn
import matplotlib.pyplot as plt

from basics import to_Var, load_model
import data_loader
import plot


# Parameters parsing
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Trained model to evaluate")
parser.add_argument("dataset", type=str, help="Dataset used for training")
parser.add_argument("subset", type=str, help="Subset to calculate the acc. on")
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
subset = args.subset


# Loads the model
model = load_model(dataset_name, model_name)


# Loads the specified subset from the specified database
images, labels = getattr(data_loader, subset)(dataset_name)


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


# Yields the indices of the first n wrong predictions.
def errors(n=len(images)):
    i = 0
    l = len(images)
    while i < l and n > 0:
        image, label = load_image(i), load_label(i)
        if prediction(image) != label:
            yield i
            n -= 1
        i += 1


# Yields the indices of the first n correct predictions.
def not_errors(n=len(images)):
    i = 0
    l = len(images)
    while i < l and n > 0:
        image, label = load_image(i), load_label(i)
        if prediction(image) == label:
            yield i
            n -= 1
        i += 1


# ATTACK FUNCTIONS
# ----------------

class Attacker(nn.Module):
    def __init__(self, p, lr):
        super(Attacker, self).__init__()
        self.p = p
        self.r = nn.Parameter(torch.zeros(1, 1, 28, 28))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return (x + self.r).clamp(0, 1)

    def loss_fn(self, image, digit):
        adv_image = self.forward(image)
        conf = model(adv_image)[0, digit]
        norm = (adv_image - image).abs().pow(self.p).sum()
        if (conf < 0.2).data[0]:
            return norm
        elif (conf < 0.9).data[0]:
            return conf + norm
        else:
            return conf - norm


def GDA(image, steps=500, p=2, lr=1e-3):
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
        print(f"Step {i:4} -- conf: {conf:0.4f}, L_{p}(r): {norm:0.20f}",
              end='\r')
        norms.append(norm)
        confs.append(conf)
    print()
    success = prediction(adv_image) != digit
    return(success, adv_image, norms, confs)


def GDA_break(image, max_steps=500, p=2, lr=1e-3):
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
        print(f"Step {steps:4} -- conf: {conf:0.4f}, L_{p}(r): {norm:0.10f}",
              end='\r')
        norms.append(norm)
        confs.append(conf)
    print()
    return(steps, adv_image, norms, confs)


def GDA_graph(image_id, steps=500, p=2, lr=1e-3):
    image = load_image(image_id)
    success, adv_image, norms, confs = GDA(image, steps, p, lr)
    plot.attack_history(norms, confs)
    plt.show()
    path = os.path.join("..", "docs", "images", "last_attack_history.png")
    plt.savefig(path, transparent=True)
    confc = lambda i: confidence(i, prediction(image))
    plot.space_exploration(image, adv_image, confc)
    plt.show()
    if success:
        print("\nAttack suceeded")
        image_pred = prediction(image)
        image_conf = 100 * confidence(image, image_pred)
        adv_image_pred = prediction(adv_image)
        adv_image_conf = 100 * confidence(adv_image, adv_image_pred)
        plot.attack_result(model_name, image_id, p,
                           image, image_pred, image_conf,
                           adv_image, adv_image_pred, adv_image_conf)
        path = os.path.join("..", "docs", "images", "last_attack_result.png")
        plt.savefig(path, transparent=True)
        plt.show()
    else:
        print("\nAttack failed")


def GDA_attack_break_graph(image_id, max_steps=500, p=2, lr=1e-3):
    image = load_image(image_id)
    success, adv_image, norms, confs = GDA_break(image, max_steps, p, lr)
    plot.attack_history(norms, confs)
    path = os.path.join("..", "docs", "images", "last_attack_history.png")
    plt.savefig(path, transparent=True)
    plt.show()
    image_pred = prediction(image)
    image_conf = confidence(image, image_pred)
    adv_image_pred = prediction(adv_image)
    adv_image_conf = confidence(adv_image, adv_image_pred)
    plot.attack_result(model_name, image_id, p,
                       image, image_pred, image_conf,
                       adv_image, adv_image_pred, adv_image_conf)
    path = os.path.join("..", "docs", "images", "last_attack_result.png")
    plt.savefig(path, transparent=True)
    plt.show()


# RESISTANCE FUNCTIONS
# --------------------

# A resistance value greater than 1000 is not possible.
# Which is why 10000 will represent infinity when the attack fails.

def resistance_N(image_id, steps):
    success, _, norms, _ = GDA(load_image(image_id), steps)
    if success:
        return norms[-1]
    return 10000


def resistance_max(image_id, steps):
    success, _, norms, _ = GDA(load_image(image_id), steps)
    if success:
        return max(norms)
    return 10000


def resistance_min(image_id, max_steps):
    steps = attack_break(load_image(image_id), max_steps)[0]
    if steps < max_steps:
        return steps
    return 10000


# Computes the N-resistance, max_resistance and min_resistance in a single pass
def resistances_3(image_id, steps):
    attack_result = GDA(load_image(image_id), steps)
    success, _, norms, confs = attack_result
    if success:
        res_N = norms[-1]
        res_max = max(norms)
        res_min = 1 + next((i for i, c in enumerate(confs) if c <= 0.2), steps)
        return (res_N, res_max, res_min)
    else:
        return (10000, 10000, 10000)


def resistances_lists(list, steps):
    L_res_N, L_res_max, L_res_min = [], [], []
    for image_id in list:
        res_N, res_max, res_min = resistances_3(image_id, steps)
        L_res_N += [res_N]
        L_res_max += [res_max]
        L_res_min += [res_min]
    return (L_res_N, L_res_max, L_res_min)


# ADVERSARIAL CORRECTIONS
# -----------------------

def labels_list(list):
    return [load_label(i) for i in list]


def pred_labels_list(list):
    return [prediction(load_image(i)) for i in list]


def corr_labels_list(list, steps):
    return [prediction(attack_break(load_image(i), steps)[1]) for i in list]


def error_count(labels, pred_labels, corr_labels, resistances, criterion):
    errs = 0
    for l, p, a, r in zip(labels, pred_labels, corr_labels, resistances):
        if r > criterion:
            errs += 1 * (p != l)
        else:
            errs += 1 * (a != l)
    return errs
