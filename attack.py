"""
Adversarial examples creation against a specified model

Syntax: python -i attack.py MODEL DATASET SUBSET
"""


import os
import sys
import torch
from torch import nn
from basics import to_Var, load_model
import data_loader
import matplotlib.pyplot as plt
import plot


# Passed parameters
model_name = sys.argv[1]
dataset_name = sys.argv[2]
subset = sys.argv[3]

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
        print("Step {:4} -- conf: {:0.4f}, L_{}(r): {:0.20f}"
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
    plt.savefig("../car-crash/docs/images/attack_{}.png".format(image_id),
                transparent=True)
    plt.show()
    if success:
        print("\nAttack suceeded")
        image_pred = prediction(image)
        image_conf = 100 * confidence(image, image_pred)
        adv_image_pred = prediction(adv_image)
        adv_image_conf = 100 * confidence(adv_image, adv_image_pred)
        plot.compare(model_name, image_id, p,
                     image, image_pred, image_conf,
                     adv_image, adv_image_pred, adv_image_conf)
        image_name = model_name + "_adv_{}.png".format(image_id)
        plt.savefig("attack_results/" + image_name, transparent=True)
        plt.show()
    else:
        print("\nAttack failed")


def attack_break_graph(image_id, max_steps=500, p=2, lr=1e-3):
    image = load_image(image_id)
    success, adv_image, norms, confs = attack_break(image, max_steps, p, lr)
    plot.attack_history(norms, confs)
    plt.savefig("../car-crash/docs/images/attack.png", transparent=True)
    plt.show()
    image_pred = prediction(image)
    image_conf = confidence(image, image_pred)
    adv_image_pred = prediction(adv_image)
    adv_image_conf = confidence(adv_image, adv_image_pred)
    plot.compare(model_name, image_id, p,
                 image, image_pred, image_conf,
                 adv_image, adv_image_pred, adv_image_conf)
    image_name = model_name + "_adv_{}.png".format(image_id, p)
    plt.savefig("attack_results/" + image_name, transparent=True)
    plt.show()


# RESISTANCE FONCTIONS
# --------------------

# A resistance value greater than 1000 is highly unlikely.
# 10000 will thus represent infinity when the attack fails.

def resistance_N(image_id, steps):
    success, _, norms, _ = attack(load_image(image_id), steps)
    if success:
        return norms[-1]
    return 10000


def resistance_max(image_id, steps):
    success, _, norms, _ = attack(load_image(image_id), steps)
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
    attack_result = attack(load_image(image_id), steps)
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


# DISCRIMINATOR DATABASE
# ----------------------


def create_discriminator_test_dataset():
    path = 'data/' + dataset_name + '/'
    empty = torch.Tensor([])
    if not os.path.exists(path + 'test_confs.pt'):
        torch.save((empty, empty.byte()), path + 'test_confs.pt')
    if not os.path.exists(path + 'test_norms.pt'):
        torch.save((empty, empty.byte()), path + 'test_norms.pt')
    all_norms, valid_preds = torch.load(path + 'test_norms.pt')
    all_confs, _ = torch.load(path + 'test_confs.pt')
    images, labels = data_loader.test(dataset_name)
    pos = len(valid_preds)
    for i in range(pos, len(images)):
        if i % 100 == 0:
            torch.save((all_norms, valid_preds), path + 'test_norms.pt')
            torch.save((all_confs, valid_preds), path + 'test_confs.pt')
            print(i)
        image = to_Var(images[i].view(1, 1, 28, 28))
        label = labels[i]
        pred_label = prediction(image)
        attack_result = attack(image, 500)
        norms = torch.Tensor(attack_result[2][::10]).view(1, 50)
        confs = torch.Tensor(attack_result[3][::10]).view(1, 50)
        valid_pred = torch.Tensor(1 * (pred_label == label)).byte()
        all_norms = torch.cat((all_norms, norms), 0)
        all_confs = torch.cat((all_confs, confs), 0)
        valid_preds = torch.cat((valid_preds, valid_pred), 0)
    torch.save((all_norms, valid_preds), path + 'test_norms.pt')
    torch.save((all_confs, valid_preds), path + 'test_confs.pt')


def create_discriminator_train_dataset():
    path = 'data/' + dataset_name + '/'
    empty = torch.Tensor([])
    if not os.path.exists(path + 'train_confs.pt'):
        torch.save((empty, empty.byte()), path + 'train_confs.pt')
    if not os.path.exists(path + 'train_norms.pt'):
        torch.save((empty, empty.byte()), path + 'train_norms.pt')
    all_norms, valid_preds = torch.load(path + 'train_norms.pt')
    all_confs, _ = torch.load(path + 'train_confs.pt')

    train_images, train_labels = data_loader.train(dataset_name)
    val_images, val_labels = data_loader.val(dataset_name)
    images = torch.cat((train_images, val_images), 0)
    labels = torch.cat((train_labels, val_labels), 0)

    pos = len(valid_preds) // 2

    for pos, (i, j) in enumerate(zip(errors(), not_errors())):
        if 2 * pos < len(valid_preds):
            pass
        else:
            if len(valid_preds) % 100 == 0:
                torch.save((all_norms, valid_preds), path + 'train_norms.pt')
                torch.save((all_confs, valid_preds), path + 'train_confs.pt')
                print(len(valid_preds))
            image_e = to_Var(images[i].view(1, 1, 28, 28))
            image_v = to_Var(images[j].view(1, 1, 28, 28))
            label_e, label_v = labels[i], labels[j]
            attack_result_e = attack(image_e, 500)
            attack_result_v = attack(image_v, 500)
            norms_e = torch.Tensor(attack_result_e[2][::10]).view(1, 50)
            confs_e = torch.Tensor(attack_result_e[3][::10]).view(1, 50)
            norms_v = torch.Tensor(attack_result_v[2][::10]).view(1, 50)
            confs_v = torch.Tensor(attack_result_v[3][::10]).view(1, 50)
            valid_pred = torch.Tensor([0, 1]).byte()
            all_norms = torch.cat((all_norms, norms_e, norms_v), 0)
            all_confs = torch.cat((all_confs, confs_e, confs_v), 0)
            valid_preds = torch.cat((valid_preds, valid_pred), 0)
    torch.save((all_norms, valid_preds), path + 'train_norms.pt')
    torch.save((all_confs, valid_preds), path + 'train_confs.pt')
