"""
Contributions
"""

import numpy as np


def get_contributors(modules, n=None, alpha=None, alpha_global=None):
    """
    For each non-initial layer in modules, get the name of the neurons that
    contribute and inhibit the neuron activation most.

    Must specify one of n or alpha.
    :param modules: a list of nn.Modules, assumed to be convolutional blocks
    whose weight matrix can be accessed directly via the .weight attribute
    :param n: number of contributing neurons to get.
    :param alpha: for each neuron get contributing neurons in this top percentile of weights FOR THE NEURON.
    :param alpha_global: for each neuron get contributing neurons in this top percentile of weights ACROSS ALL NEURONS IN THE LAYER (therefore # of weights will be variable)

    You must specify only one of either n, alpha, or alpha global.
    Specify one of either n or alpha
    """
    nones = sum([n is None, alpha is None, alpha_global is None])
    if nones != 2:
        raise ValueError("Must specify exactly one of n, alpha, or alpha_global")

    weights = [m.weight.detach().cpu().numpy() for m in modules]

    contr = [None]
    inhib = [None]
    for curr in weights[1:]:
        # curr: (in_channels x out_channels x h x w)
        # Take average or max over kernel? Let's do max
        curr = curr.mean(2).mean(2)
        if n is not None:
            raise NotImplementedError
            #  inhib_threshold = n
            #  contr_threshold = max_kernel.shape[0] - n
            #  max_kernel = np.argsort(max_kernel, axis=0)
        else:
            # Compute by threshold
            if alpha_global is not None:
                thresholds = np.quantile(curr, [alpha_global, 1 - alpha_global],
                                         keepdims=True)
            elif alpha is not None:
                thresholds = np.quantile(curr, [alpha, 1 - alpha],
                                         axis=0, keepdims=True)
            inhib_threshold = thresholds[0]
            contr_threshold = thresholds[1]

        kernel_inhib = curr < inhib_threshold
        kernel_contr = curr > contr_threshold

        inhib.append(kernel_inhib.T)
        contr.append(kernel_contr.T)

    return contr, inhib
