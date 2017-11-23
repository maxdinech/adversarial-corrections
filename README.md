# Classification et attaques de MNIST

## 1. La base de données MNIST

### MNIST

Il s'agit d'une base de donées de chiffres manuscrits : 50.000 qui servent d'entrainement, 10.000 qui servent de test et 10.000 qui servent de validation.

> :heavy_exclamation_mark: Les données de validation ne sont à utiliser qu'à la toute fin, par souci d'honnêteté intellectuelle. (Pour éviter de faire de l'overfitting des HP sur les images de validations !)

Les images sont des tenseurs de la forme `1x28x28` (le 1 représente les couches de couleur), et les labels sont représentés par des scalaires entiers. La base de données est contenue dans les fichiers `train.py`, `test.py` et `val.py`, chacun sous la forme d'un couple de tenseurs `(images, labels)`.

### Importation avec mnist_loader.py

Ce fichier permet de créer automatiquement si nécessaire et d'importer facilement les bases de données `train`, `test` et `val`, de la manière suivante :

```Python
images, labels = mnist_loader.train(nb_éléments)
```

Même syntaxe pour `test` et `val`



## 2. Définition et entrainement des modèles

### Description des modèles

#### MLP à deux couches cachées (`MLP` et `MLP_d`)

Couches :

- **fc:** 748 -> 128 (ReLU)
- (**Dropout:** p=0.4)
- **fc:** 128 -> 128 (ReLU)
- (**Dropout:** p=0.4)
- **fc:** 128 -> 10 (softmax)

Hyperparamètres :

- Epochs : 30
- batch_size = 32
- Optimiseur : Adam(lr = 3e-4)

#### CNN à deux convolutions (`CNN` et `CNN_d`)

![CNN à deux convolution](../docs/images/CNN2_small.png)

Couches :

- **Conv:** noyau 5x5, 1 -> 20 layers
- **MaxPool:** noyau 2x2
- **Conv:** noyau 3x3, 20 -> 40 layers
- **MaxPool:** noyau 2x2
- (**Dropout:** p=0.4)
- **fc:** 5x5x40 -> 120 (ReLU)
- (**Dropout:** p=0.4)
- **fc:** 120 -> 10 (softmax)

Hyperparamètres :

- Epochs : 30
- batch_size = 32
- Optimiseur : Adam(lr = 3e-4)


### Définition des modèles : `architectures.py`

Les modèles utilisés sont entièrement définis par leurs classes, dans `architectures.py`.

On ajoute à chaque modèle ses hyperparamètres (`lr`, `epochs` et `batch_size`), sa fonction d'erreur (`loss_fn`) et son optimiseur (`optimizer`), ce qui permet de jongler beaucoup plus facilement entre les modèles.


### Entrainement des modèles : `train.py`

Le fichier `train.py` prend en paramètre le nom de la classe de modèle à entraîner, et un booléen qui décide de l'enregistrement du modèle au format `.pt` dans `modeles/`. Par exemple la commande suivante :

    python train.py CNN True

### Résultats obtenus

|   Réseau |  MLP   | MLP_d  |  CNN   | CNN_d  |
|---------:|:------:|:------:|:------:|:------:|
|      acc | 98.79% | 97.88% | 99.64% | 99.35% |
| test_acc | 97.23% | 97.24% | 98.95% | 99.10% |


## 2. Attaques adversaires

### Description des attaques

...

### Résultats obtenus

...

### Évaluer la faiblesse d'un réseau

-> On trace les distribution des normes minimales pour perturber chaque image dans une même base de données, selon des normes diférentes. On compare les résultats obtenus contre différents réseaux.