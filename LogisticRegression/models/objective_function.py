#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Logistic Regression with a Non-convex Regularizer
"""

# Import packages
import numpy as np


# Create custom objective and Augmented Lagrangian functions
class RegularizedLogisticRegression:
    '''
    Non-convex regularized logistic regression function from the above mentioned paper
    '''

    def __init__(self, alpha, beta, num_nodes):
        '''
        Initialize the regularization terms
        '''

        self.alpha = alpha
        self.beta = beta
        self.num_nodes = num_nodes

    def forward(self, data, labels, weights):
        '''
        Compute the Logistic Loss given the data
        '''

        # Gather problem dimensions
        N = data.shape[0]

        # Compute product
        product = np.multiply(labels, np.matmul(data, weights))

        # Compute LogReg
        f = (1 / N) * np.sum(np.log(1 + np.exp(-product)))

        # Nonconvex Regularizer
        row_square = np.square(weights)
        division = self.beta * np.divide(self.alpha * row_square, 1 + self.alpha * row_square)
        reg = np.sum(division)

        # Compute objective
        objective = (1 / self.num_nodes) * (f + reg)

        return objective

    def gradient(self, data, labels, weights):
        '''
        Computes the gradient at a given point with respect to the weights
        '''

        # Gather problem dimensions
        N = data.shape[0]

        # Numerator and denominator
        product = np.multiply(labels, np.matmul(data, weights.transpose()))
        denominator = 1 + np.exp(product)
        numerator = np.multiply(-labels[:, np.newaxis], data)
        try:
            f_grad = (1 / N) * np.sum(np.divide(numerator, denominator.T), axis=0)
        except:
            f_grad = (1 / N) * np.sum(np.divide(numerator, denominator.T[:, np.newaxis]), axis=0)

        # Nonconvex Regularizer
        row_square = np.square(weights)
        reg_grad = self.alpha * self.beta * np.divide(2 * weights, np.square(1 + self.alpha * row_square))

        # Combine the two
        grad = (1 / self.num_nodes) * (f_grad + reg_grad)

        return grad


# Create custom Augmented Lagrangian function
class AugmentedLagrangian:
    '''
    Augmented Lagrangian implementation from the above paper - in terms of the primal variable X
    L(x,x_0,y) = f(x) + <y,x-x_0> + (1/2*eta)||x-x_0||^2
    '''

    def __init__(self, function, eta):
        '''
        Initialize the method
        '''

        # Make sure function has appropriate methods
        if hasattr(function, 'forward') & hasattr(function, 'gradient'):
            self.function = function

        else:
            print('[Error] Please upload a valid function class.')
            return

        # Initialize the penalty parameter
        self.eta = eta

    def forward(self, data, labels, weights, global_weights, dual_variable):
        '''
        Compute objective value
        '''

        # Compute function value
        f = self.function.forward(data, labels, weights)

        # Compute extra_ring terms
        dual = np.inner(dual_variable, weights - global_weights)
        penalty = (1 / (2 * self.eta)) * np.linalg.norm(weights - global_weights)**2

        # Compute objective
        lagrangian_objective = f + dual + penalty

        return lagrangian_objective

    def gradients(self, data, labels, weights, global_weights, dual_variable):
        '''
        Compute gradient of the objective (w.r.t. each variable)
        '''

        # Compute gradient with respect to dual
        dual_grad = weights - global_weights

        # Compute function gradient
        f_grad = self.function.gradient(data, labels, weights)
        f_grad += dual_variable + (1 / self.eta) * dual_grad

        return f_grad

