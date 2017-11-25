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


# Returns a Variable containing `tensor`, on the GPU if CUDA is available.
def to_Var(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, requires_grad=requires_grad)


# Loads the model
try:
    model = torch.load("models/" + model_name + ".pt",
                       map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        model = model.cuda()
except FileNotFoundError:
    print("No model found")


# Plots and saves the comparison graph
def compare(img_id, image, adv_image, p):
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
              .format(model_name, image_pred, image_prob))
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(r.data.view(28, 28).cpu().numpy(), cmap='RdBu')
    plt.title("Perturbation : $\\Vert r \\Vert_{{{}}} = {:0.4f}$"
              .format(p, round(norm, 3)))
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(adv_image.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img+r)}} = {} \\small{{({:0.0f}\\%)}}"
              .format(model_name, adv_image_pred, adv_image_prob))
    plt.axis('off')
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.05)
    image_name = model_name + "_adv{}_n{}.png".format(img_id, p)
    plt.savefig("docs/images/attack_results" + image_name, transparent=True)
    plt.show()


# Plots an image (Variable)
def plot_image(image):
    plt.imshow(image.data.view(28, 28).numpy(), cmap='gray')
    plt.show()


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
        print("N = {:5.5f}  ->  step {:4}  -  conf: {:0.4f}, norm_{}(r): {:0.4f}"
              .format(N, i, conf, p, N_), end='\r')
        if prediction(adv_image).data[0] != digit:
            break
    return (i < 999), image, adv_image


# Searches the minimal div value that fools the network
def minimal_attack_dichotomy(img_id, p=2, a=0, b=4, lr=0.005):
    while b-a >= 0.001:
        c = (a+b)/2
        print()
        success, image, adv_image = attack_fixed_norm(img_id, c, p, lr)
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
            return conf + norm
        elif (conf <= 0.4).data[0]:
            return conf + norm
        else:
            return conf


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
        norms.append(N)
        probs.append(conf)
    adv_image = attacker.forward(image)
    t = list(range(500))
    plt.plot(t, norms, 'r', t, probs, 'b')
    if prediction(adv_image).data[0] != digit:
        print("\nAttack suceeded")
        plt.show()
        compare(img_id, image, adv_image, p)
    else:
        print("\nAttack failed")


# Run multiple attacks
def attacks():
    for img_id in range(10):
        for p in [2, 3, 5, 10]:
            attack(img_id, p)
