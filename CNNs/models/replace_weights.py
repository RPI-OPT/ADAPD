#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Custom optimizer which replaces a model's parameters with new parameters
        - this ensures that updates are being done properly
"""

# Import packages
import torch
import torch.optim


# Optimizer that doesnt actually do anything other than reassign values to what you want them to be
class Opt(torch.optim.Optimizer):
    '''
    Reassign model parameter values
    '''

    def __init__(self, params, lr):

        # Verify inputs
        if lr is None or lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        # Make accessible dictionary for updating
        defaults = dict(lr=lr)

        # Make super class
        super(Opt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, new_weights, device, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # Closure could be used outside of the training loop to reevaluate model in place
        # Issue is the parameters do not change unless we change them here...
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Loop over the parameters
        for group in self.param_groups:

            # Update the weights
            for ind, p in enumerate(group['params']):

                with torch.no_grad():
                    # Update the weights
                    p.data = new_weights[ind].to(device)

        return loss