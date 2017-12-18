import matplotlib.pyplot as plt
from matplotlib import rcParams


# Plots an image (Variable)
def plot_image(image):
    plt.imshow(image.data.view(28, 28).numpy(), cmap='gray')
    plt.show()


# Plots and saves the comparison graph
def compare(model_name, img_id, p,
            image, image_pred, image_conf,
            adv_image, adv_image_pred, adv_image_conf):
    model_name_tex = model_name.replace('_', '\\_')
    r = (adv_image - image)
    norm = r.norm(p).data[0]
    # Matplotlib settings
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['axes.titlepad'] = 10
    rcParams['font.family'] = "serif"
    rcParams['font.serif'] = "cm"
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
    image_name = model_name + "_adv{}_n{}.png".format(img_id, p)
    plt.savefig("attack_results/" + image_name, transparent=True)


def norm_and_conf(norms, confs):
    t = list(range(len(norms)))
    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['font.serif'] = "cm"
    rcParams['font.size'] = 12
    plt.plot(t, norms, 'r', t, confs, 'b')
    plt.legend(["$Pred$", "$Conf_c$"])
