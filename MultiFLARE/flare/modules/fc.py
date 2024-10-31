# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj, Kelian Baert

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


class Sine(nn.Module):
    r"""Applies the sine function with frequency scaling element-wise:

    :math:`\text{Sine}(x)= \sin(\omega * x)`

    Args:
        omega: factor used for scaling the frequency

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

def make_module(module):
    # Create a module instance if we don't already have one
    if isinstance(module, torch.nn.Module):
        return module
    else:
        return module()


class FullyConnectedBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, activation=torch.nn.ReLU):
        super().__init__()

        self.linear = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = make_module(activation) if activation is not None else torch.nn.Identity()

    def forward(self, input):
        return self.activation(self.linear(input))

def siren_init_first(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-1 / n, 
                                     1 / n)

def siren_init(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    omega = kwargs['omega']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-np.sqrt(6 / n) / omega, 
                                     np.sqrt(6 / n) / omega)

def init_weights_normal(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)

def init_weights_normal_last(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight, gain=1)
            module.weight.data = -torch.abs(module.weight.data)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)
            
def init_weights_zero(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.zeros_(module.weight)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)

class FC(nn.Module):
    def __init__(self, features: List[int], activation='relu', last_activation=None, bias=True, first_omega=30, hidden_omega=30.0, per_layer_in_out: List[Tuple[int, int]]=None):
        super().__init__()

        activations_and_inits = {
            'sine': (Sine(first_omega),
                     siren_init,
                     siren_init_first,
                     None),
            'relu': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     init_weights_normal),
            'relu2': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     init_weights_normal_last),
            'softplus': (nn.Softplus(),
                     init_weights_normal,
                     None,
                     None),
        }

        if activation is None:
            activation_fn, weight_init, first_layer_init, last_layer_init  = None, init_weights_zero, init_weights_zero, init_weights_zero
        elif isinstance(activation, torch.nn.Module):
            activation_fn, weight_init, first_layer_init, last_layer_init  = activation, init_weights_normal, init_weights_normal, init_weights_normal
        else:
            activation_fn, weight_init, first_layer_init, last_layer_init = activations_and_inits[activation]

        if per_layer_in_out is None:
            per_layer_in_out = [(features[i-1], features[i]) for i in range(1, len(features))]

        layers = []
        for i, (n_in, n_out) in enumerate(per_layer_in_out):
            if i == 0:
                layer_init = first_layer_init
                layer_activation = activation_fn
            elif i == len(per_layer_in_out) - 1:
                layer_init = last_layer_init
                layer_activation = last_activation
            else:
                layer_init = weight_init
                layer_activation = activation_fn

            layer = FullyConnectedBlock(n_in, n_out, bias=bias, activation=layer_activation)
            # initialize the weights
            if layer_init is not None:
                layer.apply(lambda module: layer_init(module=module, n=n_in, omega=hidden_omega))

            layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
