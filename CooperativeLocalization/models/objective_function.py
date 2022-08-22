#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Cooperative Localization
"""

# Import packages
import numpy as np


# Create Custom Objective Function
class CooperativeLocalization:
    '''
    Non-convex cooperative localization from the above paper
    '''

    def __init__(self, num_nodes, prob_dim, num_targets):

        # Extract problem information
        self.num_nodes = num_nodes
        self.problem_dimension = prob_dim
        self.num_targets = num_targets

    def forward(self, agent_loc, agent_noises, target_guesses):

        # Compute objective
        objective = 0
        for i in range(self.num_targets):
            objective += (1 / (4 * self.num_nodes)) * (agent_noises[i] - np.linalg.norm(target_guesses[i*self.problem_dimension:(i+1)*self.problem_dimension] - agent_loc)**2)**2

        return objective

    def gradient(self, agent_loc, agent_noises, target_guesses):
        '''Computes the gradient at a given point with respect to the weights'''

        # Get problem dimension and number of targets

        # Compute gradient
        grad = np.zeros(shape=(self.problem_dimension * self.num_targets, ))

        for i in range(self.num_targets):
            grad[i*self.problem_dimension:(i+1)*self.problem_dimension] = - (1 / self.num_nodes) * (target_guesses[i*self.problem_dimension:(i+1)*self.problem_dimension] - agent_loc) * (agent_noises[i] - np.linalg.norm(target_guesses[i*self.problem_dimension:(i+1)*self.problem_dimension] - agent_loc)**2)

        return grad


# Create Custom Augmented Lagrangian Function
class AugmentedLagrangian:
    '''
    Augmented Lagrangian implementation from the above paper - only involving terms about X

    L(x,x_0,y) = f(x) + <y,x-x_0> + (1/2*eta)||x-x_0||^2
    '''

    def __init__(self, function, eta):

        # Make sure function has appropriate methods
        if hasattr(function, 'forward') & hasattr(function, 'gradient'):
            self.function = function

        else:
            print('[Error] Please upload a valid function class.')
            return

        # Initialize the penalty parameter
        self.eta = eta

    def forward(self, agent_loc, agent_noise, target_guess, global_weights, dual_variable):
        '''
        Compute objective value
        '''

        # Compute function value
        f = self.function.forward(agent_loc, agent_noise, target_guess)

        # Compute extra_ring terms
        dual = np.inner(dual_variable, target_guess - global_weights)
        penalty = (1 / (2 * self.eta)) * np.linalg.norm(target_guess - global_weights)**2

        # Compute objective
        lagrangian_objective = f + dual + penalty

        return lagrangian_objective

    def gradients(self, agent_loc, agent_noise, target_guess, global_weights, dual_variable):
        '''
        Compute gradient of the objective (w.r.t. each variable)
        '''

        # Compute gradient with respect to dual
        dual_grad = target_guess - global_weights

        # Compute function gradient
        f_grad = self.function.gradient(agent_loc, agent_noise, target_guess)
        f_grad += dual_variable + (1 / self.eta) * dual_grad

        return f_grad
