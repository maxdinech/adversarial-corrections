# Building, attacking and improving classifiers for MNIST

The aim of this school project is to show the vulnerabilities of classifier networks to adverse examples and to explore different protection techniques. 

## Getting Started

### 1. Creating the MNIST dataset

The MNIST dataset is automatically downloaded and divided into three parts : train.pt, containing 50.000 samples, test.pt and val.pt containing 10.000 samples each.

Originally, when loading MNIST, PyTorch divieds the dataset between train.pt and test.pt, but a third file val.pt allows to test a model performance after fitting its hyperparameters using the test dataset.

### 2. Network training

The networks are defined in `architectures.py`. For the moment, the following networks are available : a MLP, a CNN, and their dropout versions (MLP_d and CNN_d). Other networks will be added.

![CNN with 2 convolutions](../docs/images/CNN_small.png)

*The CNN model.*

Note that the training parameters (`lr`, `epochs` and `batch_size`) and functions (`loss_fn` and `optimizer`) are included in the class definition of the model: it makes switching between models easier and makes it possible to use a universal training file : `train.py`

For example, to train the CNN model with dropout, you just need to run:

```
python train.py CNN True
```

The second parameter specifies wether the model will be saved (in `models/CNN_d.pt` if it is the case).

#### Some results:

|  Network |  MLP   | MLP_d  |  CNN   | CNN_d  |
|---------:|:------:|:------:|:------:|:------:|
|      acc | 98.79% | 97.88% | 99.64% | 99.35% |
| test_acc | 97.23% | 97.24% | 98.95% | 99.10% |

## 3. Network attacks

### Different ways to attack the network

The attacks are somewhat similar to a network training: we make a gradient descent on a 28x28 Variable `r` so that model.forward(image + r) gives a wrong prediction. More formally:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\small&space;\begin{cases}&space;\Vert&space;r&space;\Vert_p&space;=&space;norm\\&space;Img&space;&plus;&space;r&space;\in&space;[0,&space;1]\\&space;Pred(img&plus;r)&space;\neq&space;Pred(Img)\\&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\small&space;\begin{cases}&space;\Vert&space;r&space;\Vert_p&space;=&space;norm\\&space;Img&space;&plus;&space;r&space;\in&space;[0,&space;1]\\&space;Pred(img&plus;r)&space;\neq&space;Pred(Img)\\&space;\end{cases}" title="\small \begin{cases} \Vert r \Vert_p = norm\\ Img + r \in [0, 1]\\ Pred(img+r) \neq Pred(Img)\\ \end{cases}" /></a>

A network attack takes in parameters the `id` of the image to attack, the euclidean norm `p` used to determine the norm of the perturbation `r`, the norm value of the perturbation that will be used during the attack, and the learning rate of the gradient descent.


To attack a previously trained and saved model, load the attack.py file, for instance:

```
python -i attack.py CNN_d
```

Multiple functions are then available.

- `attack()` 

---

## Project requirements

- Python 3
- PyTorch
- numpy
- matplotlib
- texlive, ghostscript and dvipng (for a fancy matplotlib latex-style prining)
- tqdm