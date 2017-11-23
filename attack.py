"""
Adversarial examples creation against a specified model

Syntax: python -i attack.py CNN
"""


import sys
import torch
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
def compare(image1, r, image2, img_id, p, div):
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['axes.titlepad'] = 10
    rcParams['font.family'] = "serif"
    rcParams['font.serif'] = "cm"
    fig = plt.figure(figsize=(7, 2.5), dpi=200)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image1.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img)}} = {}"
              .format(model_name, prediction(image1)))
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(r.data.view(28, 28).cpu().numpy(), cmap='RdBu')
    plt.title("Perturbation : $\\Vert r \\Vert_{{{}}} = {}$"
              .format(p, round(div, 3)))
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image2.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img + r)}} = {}"
              .format(model_name, prediction(image2)))
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


# Loads the #img_id image from the train database
def load_image(img_id):
    images, _ = mnist_loader.train(img_id+1)
    return to_Var(images[img_id].view(1, 1, 28, 28))


# Returns the class prediction of the model on an image (Variable)
def prediction(image):
    return model.forward(image).max(1)[1].data[0]


# Network attack
def attack(img_id, lr=0.005, div=0.2, p=2):
    image = load_image(img_id)
    chiffre = prediction(image)
    r = to_Var(torch.zeros(1, 1, 28, 28), requires_grad=True)
    adv = lambda image, r: (image + (r * div / (1e-5 + r.norm(p)))).clamp(0, 1)
    # the 1-e5 added to the norm prevents r from diverging to nan
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


# Searches the minimal div value that fools the network
def minimal_attack(img_id, p=2, a=0, b=4, lr=0.005):
    while b-a >= 0.001:
        c = (a+b)/2
        print("\n\nImage {} : norm_{}(r) = {}\n".format(img_id, p, c))
        succes, img, r, adv = attack(img_id, lr, c, p)
        if succes:
            b = c
            img_local, r_local, adv_local = img, r, adv
        else:
            a = c
    print("\n\nMinimal div value: ", b)
    compare(img_local, r_local, adv_local, img_id, p, b)


# Run multiple attacks
def attacks():
    for img_id in range(10):
        for p in [2, 3, 5, 10]:
            minimal_attack(img_id, p, 0, 5)
