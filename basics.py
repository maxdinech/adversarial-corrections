"""
Basic PyTorch functions
"""

import torch
from torch.autograd import Variable


# Returns a Variable containing `tensor`, on the GPU if CUDA is available.
def to_Var(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, requires_grad=requires_grad)
