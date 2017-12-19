"""
Basic PyTorch functions
"""


import torch
from torch.autograd import Variable
import architectures


# Returns a Variable containing `tensor`, on the GPU if CUDA is available.
def to_Var(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, requires_grad=requires_grad)


def load_architecture(model_name):
    model = getattr(architectures, model_name)()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_model(model_name):
    try:
        model = torch.load("models/" + model_name + ".pt",
                           map_location=lambda storage, loc: storage)
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    except FileNotFoundError:
        print("No trained model found.")
        print("run : python train.py", model_name, "to train the model.")
