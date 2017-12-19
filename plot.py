"""
Plotting functions used in attack.py
"""


import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['font.family'] = "serif"
rcParams['font.serif'] = "cm"


# Plots an image (Variable)
def plot_image(image):
    plt.imshow(image.data.view(28, 28).numpy(), cmap='gray')


# Plots and saves the comparison graph of an adversarial image
def compare(model_name, img_id, p,
            image, image_pred, image_conf,
            adv_image, adv_image_pred, adv_image_conf):
    model_name_tex = model_name.replace('_', '\\_')
    r = (adv_image - image)
    norm = r.norm(p).data[0]
    # Matplotlib settings
    rcParams['axes.titlepad'] = 10
    rcParams['font.size'] = 8
    fig = plt.figure(figsize=(7, 2.5), dpi=180)
    # Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img)}} = {} \\small{{({:0.0f}\\%)}}"
              .format(model_name_tex, image_pred, image_conf))
    plt.axis('off')
    # Perturbation
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(r.data.view(28, 28).cpu().numpy(), cmap='RdBu')
    plt.title("Perturbation : $\\Vert r \\Vert_{{{}}} = {:0.4f}$"
              .format(p, round(norm, 3)))
    plt.axis('off')
    # Adversarial image
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(adv_image.data.view(28, 28).cpu().numpy(), cmap='gray')
    plt.title("\\texttt{{{}(img+r)}} = {} \\small{{({:0.0f}\\%)}}"
              .format(model_name_tex, adv_image_pred, adv_image_conf))
    plt.axis('off')
    # Save and plot
    fig.tight_layout(pad=1)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.05)


# Plots the history of a model training
def training_history(train_accs, test_accs):
    rcParams['font.size'] = 12
    t = list(range(len(train_accs)))
    plt.plot(t, train_accs, 'r')
    plt.plot(t, test_accs, 'b')
    plt.title("Network training history")
    plt.legend(["train accuracy", "test accuracy"])


# Plots the history of an attack
def attack_history(norms, confs):
    rcParams['font.size'] = 12
    t = list(range(len(norms)))
    plt.plot(t, norms, 'r')
    plt.plot(t, confs, 'b')
    plt.legend(["$Pred$", "$Conf_c$"])
