#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Decentralized Primal-Dual Framework for Non-convex Smooth Consensus Optimization
    Paper: https://arxiv.org/abs/2107.11321
    Code provided by: Gabe Mancino-Ball

    Problem: Logistic Regression with a Non-convex Regularizer
"""

# Import packages
import math
import time
import numpy as np
from models.objective_function import AugmentedLagrangian as AL
from models.objective_function import RegularizedLogisticRegression as RLR


# New ADAPD class
class ADAPD:
    '''
    Proposed class for solving decentralized nonconvex consensus problems.

    :param params: LIST containing local AL parameters (maximum iterations, stopping tolerance, penalty parameter, regularizers, etc.)
    :param mixing_matrix: nxn mixing matrix (symmetric, doubly stochastic)
    :param data: Nxp matrix of features (N = n * D, where D is the number of samples on each node)
    :param labels: NxNone matrix of labels corresponding to the feature inputs
    '''

    def __init__(self, params, mixing_matrix, data, labels, init_point):

        # Gather mixing matrix and number of nodes
        self.mixing_matrix = mixing_matrix
        self.num_nodes = self.mixing_matrix.shape[0]

        # Gather data information
        # > Number of local samples
        # > Dimension of the problem
        self.data = data
        self.labels = labels
        self.num_samples = int(len(self.labels) / self.num_nodes)
        self.problem_dim = self.data.shape[1]

        # Gather appropriate parameters for local agents
        if 'local_max_iters' in params:
            self.local_max_iters = params['local_max_iters']
        else:
            self.local_max_iters = 10
        if 'stopping_tol' in params:
            self.stopping_tol = params['stopping_tol']
        else:
            self.stopping_tol = 1e-16
        if 'comm_rounds' in params:
            self.comm_rounds = params['comm_rounds']
        else:
            self.comm_rounds = 1
        if 'og' in params:
            self.og = params['og']
        else:
            self.og = False
        if 'eta' in params:
            self.eta = params['eta']
        else:
            self.eta = 1e-3
        if 'beta' in params:
            self.beta = params['beta']
        else:
            self.beta = 1e-1
        if 'alpha' in params:
            self.alpha = params['alpha']
        else:
            self.alpha = 1.0
        if 'local_steps' in params:
            self.local_steps = params['local_steps']
        else:
            self.local_steps = 1e-4
        if 'dim' in params:
            self.dim = params['dim']
        else:
            self.dim = False
        if 'dim_stop_tol' in params:
            self.dim_stop_tol_base = params['dim_stop_tol']
        else:
            self.dim_stop_tol_base = self.stopping_tol
        if 'dim_power' in params:
            self.dim_power = params['dim_power']
        else:
            self.dim_power = 2
        if 'mini_batch' in params:
            self.mini_batch = params['mini_batch']
        else:
            self.mini_batch = self.num_samples

        # Initialize the objective function
        self.objective_function = RLR(self.alpha, self.beta, self.num_nodes)

        # Initialize the Augmented Lagrangian
        self.augmented_lagrangian = AL(self.objective_function, self.eta)

        # Primal local variables (X)
        self.local_variables = init_point.copy()

        # Primal consensual variables (X0)
        self.global_variables = np.zeros(shape=self.local_variables.shape)

        # Initialize the dual variables (Y, Z)
        self.local_dual_variables = np.zeros(shape=self.local_variables.shape)
        self.consensual_dual_variables = np.zeros(shape=self.local_variables.shape)
        self.old_consensual_dual_variables = np.zeros(shape=self.local_dual_variables.shape)

        # Save relevant information from each iteration
        self.consensus_violation = []
        self.norm_hist = []
        self.obj_hist = []
        self.training_accuracy = []
        self.testing_accuracy = []
        self.total_optimality = []
        self.num_grads_evaluated = 0

        # Get Chebyshev info!
        lam2 = np.sort(np.linalg.eig(self.mixing_matrix)[0])[self.num_nodes - 2]
        self.mix_step = (1 - np.sqrt(1 - lam2 ** 2)) / (1 + np.sqrt(1 - lam2 ** 2))
        self.cheby_step = np.linalg.norm(self.mixing_matrix - np.ones((self.num_nodes, self.num_nodes)) / self.num_nodes, ord=2)

        # Append first errors
        violation = np.linalg.norm(np.matmul((1 / self.num_nodes) * np.ones(shape=self.mixing_matrix.shape),
                                             self.local_variables.transpose()).transpose() - self.local_variables,
                                   ord='fro') ** 2
        norms = self.objective_function.gradient(self.data, self.labels,
                                                 (1 / self.local_variables.shape[1]) * np.sum(self.local_variables,
                                                                                              axis=1))

        self.consensus_violation.append(violation)
        self.norm_hist.append(np.linalg.norm(norms) ** 2)
        self.total_optimality.append(np.linalg.norm(norms) ** 2 + violation)

    def subsolver(self, data, labels, init_guess, current_global_variable, current_dual_variable, max_it, step_size):
        '''FISTA subsolver for ADAPD X-subproblem'''

        # Keep track of local gradients and objective history
        al_grad_norm_hist = []

        # Save initial guess
        weights = np.array([init_guess], dtype=float).flatten()

        # USE SGD IF STOCHASTIC
        if self.mini_batch != data.shape[0]:

            # For loop over iterations
            for i in range(int(max_it)):

                # Get an index
                indx = np.random.choice([t for t in range(data.shape[0])], size=self.mini_batch,
                                        replace=False).tolist()

                # Get the gradients
                al_grad = self.augmented_lagrangian.gradients(data[indx, :], labels[indx], weights, current_global_variable,
                                            current_dual_variable)

                # Check norm
                grad_norm = np.linalg.norm(al_grad) ** 2
                if grad_norm <= self.stopping_tol:
                    return weights

                # Save the outputs
                al_grad_norm_hist.append(grad_norm)

                # Update the guess
                weights = weights - (step_size) * al_grad
                weights = np.array([weights], dtype=float).flatten()

            return weights

        # USE FISTA
        else:

            # Start FISTA weights
            t0 = 1

            # Save initial guess
            extracted_point = np.array([init_guess], dtype=float).flatten()

            # Perform loop
            for i in range(int(max_it)):

                # Format and save old weights
                old_weights = weights.copy()

                # Compute momentum
                t1 = (1 + math.sqrt(1 + 4 * t0 ** 2)) / 2
                w = (t0 - 1) / t1
                t0 = t1

                # Get the gradients
                al_grad = self.augmented_lagrangian.gradients(data, labels, extracted_point, current_global_variable,
                                            current_dual_variable)

                # Check norm
                grad_norm = np.linalg.norm(al_grad) ** 2

                if grad_norm <= self.stopping_tol:
                    # print(f'Number of gradients evaluated: {len(al_grad_norm_hist)}')
                    return weights, len(al_grad_norm_hist)

                # Update the guess
                weights = extracted_point - (step_size) * al_grad
                extracted_point = weights + w * (weights - old_weights)

                # Change format bc numpy is particular
                extracted_point = np.array([extracted_point], dtype=float).flatten()
                weights = np.array([weights], dtype=float).flatten()
                # print(f'Number of gradients evaluated: {len(al_grad_norm_hist)}')

                # ONLY APPEND IF A STEP IS TAKEN
                al_grad_norm_hist.append(grad_norm)

            return weights, len(al_grad_norm_hist)

    def solve(self, outer_iterations, testing_data, testing_labels):
        '''
        Solve the decentralized problem with the proposed ADAPD method
        '''

        # Time the algorithm
        time0 = time.time()
        avg_local_time = []

        # Run the outer loop (i.e. communication rounds)
        for i in range(outer_iterations):

            # Save norms, objective, accuracy, and time
            obj = 0
            acc = 0
            local_time = []
            avg_num_grads = []

            # Check stopping tolerance is diminishing
            if self.dim:
                self.stopping_tol = self.dim_stop_tol_base / (self.num_nodes * math.pow(i + 1, self.dim_power))
            else:
                pass

            # Do local subsolvers
            for j in range(self.num_nodes):

                # Time the local node
                local_time_j = time.time()

                # Update the local solution either by multiple steps or by one step
                if self.og:
                    self.local_variables[:, j] = self.global_variables[:, j].copy() - self.eta * (
                                self.augmented_lagrangian.function.gradient(
                                    self.data[j * self.num_samples:(j + 1) * self.num_samples, :],
                                    self.labels[j * self.num_samples:(j + 1) * self.num_samples],
                                    self.local_variables[:, j]) + self.local_dual_variables[:, j].copy())

                    avg_num_grads.append(1)

                else:
                    self.local_variables[:, j], num_grads = self.subsolver(self.data[j * self.num_samples:(j + 1) * self.num_samples, :],
                    self.labels[j * self.num_samples:(j + 1) * self.num_samples], self.local_variables[:, j], self.global_variables[:, j], self.local_dual_variables[:, j], self.local_max_iters, self.local_steps)

                    avg_num_grads.append(num_grads)

                # Compute objective value
                obj += self.augmented_lagrangian.function.forward(
                    self.data[j * self.num_samples:(j + 1) * self.num_samples, :],
                    self.labels[j * self.num_samples:(j + 1) * self.num_samples], self.local_variables[:, j])

                # Train on each node
                pred_j = np.sign(np.matmul(self.data[j*self.num_samples:(j+1)*self.num_samples, :], self.local_variables[:, j]))
                acc += np.sum(pred_j == self.labels[j*self.num_samples:(j+1)*self.num_samples]) / self.num_samples

                # Append time
                local_time.append(round(time.time() - local_time_j, 5))

            # Update gradient information
            self.num_grads_evaluated += int(sum(avg_num_grads) / len(avg_num_grads))

            # Update X0
            self.global_variables = 0.5 * (self.local_variables.copy() + self.global_variables.copy() + self.eta * (
                self.local_dual_variables.copy() - 2 * self.consensual_dual_variables.copy() + self.old_consensual_dual_variables.copy()
            ))

            # Update Y
            self.local_dual_variables = self.local_dual_variables + \
                (1 / self.eta) * (self.local_variables - self.global_variables)

            # Update Z - communication here!
            self.old_consensual_dual_variables = self.consensual_dual_variables.copy()
            if self.comm_rounds == 1:
                self.consensual_dual_variables = self.consensual_dual_variables + \
                                             (1 / self.eta) * np.matmul(np.eye(self.num_nodes) - self.mixing_matrix, self.global_variables.transpose().copy()).transpose()
            else:
                # Either Chebyshev acceleration
                mixed_global = self.chebyshev(self.global_variables, self.comm_rounds, self.cheby_step)
                self.consensual_dual_variables = self.consensual_dual_variables + \
                                                 (1 / self.eta) * (self.global_variables.copy() - mixed_global.copy())

            # Consensus violation
            violation = np.linalg.norm(np.matmul((1 / self.num_nodes) * np.ones(shape=self.mixing_matrix.shape),
                self.local_variables.transpose()).transpose() - self.local_variables, ord='fro')**2

            # Append
            self.consensus_violation.append(violation)

            # Update norms and objective history
            norms = self.objective_function.gradient(self.data, self.labels, (1 / self.local_variables.shape[1]) * np.sum(self.local_variables, axis=1))
            self.norm_hist.append(np.linalg.norm(norms) ** 2)
            self.obj_hist.append(obj / self.num_nodes)
            self.training_accuracy.append(acc / self.num_nodes)
            self.total_optimality.append(np.linalg.norm(norms) ** 2 + violation)

            # Save average training time on local machine
            avg_local_time.append((1 / len(local_time)) * sum(local_time))

            # Test the method
            test_acc = self.test(self.local_variables, testing_data, testing_labels)
            self.testing_accuracy.append(test_acc)

            if i % 100 == 0:
                print(f'[ADAPD] Testing accuracy is: {test_acc} %.')

        # Record time
        time1 = time.time() - time0

        print(f'[ADAPD] Training completed in {round(time1, 4)} seconds. '
              f'Average time spent on each node: {(1 / len(avg_local_time)) * sum(avg_local_time)} seconds. '
              f'Number of gradients evaluated: {self.num_grads_evaluated}.\n')

    def test(self, weights, testing_data, testing_labels):
        '''Test the AVERAGE weights on new data points'''

        avg_weights = (1 / weights.shape[1]) * np.sum(weights, axis=1)
        pred = np.sign(np.matmul(testing_data, avg_weights))
        acc = np.sum(pred == testing_labels) / (len(testing_labels))

        return round(acc, 2)

    def chebyshev(self, x, iterates, step_size):
        '''
        Perform Chebyshev accelerated gossip

        Algorithm 6.1 in:
        https://www.asc.tuwien.ac.at/~winfried/teaching/106.079/SS2017/downloads/iter.pdf
        '''

        # Save old information and begin of mu
        old_x = x.copy()
        mu0 = 1
        mu1 = (1 / self.cheby_step)

        # Do first update
        x = np.matmul(self.mixing_matrix, x.transpose().copy()).transpose()

        # Run for loop - since the first iterate of Chebyshev does a communication,
        # we require that we perform `range(iterates - 1)` iterations since
        # iterates = 1 corresponds to no for loop
        for i in range(iterates - 1):

            # Save information and update mu
            curr_x = x.copy()
            new_mu = (2 / step_size) * mu1 - mu0

            # Do the update
            x = (2 / step_size) * (mu1 / new_mu) * np.matmul(self.mixing_matrix, curr_x.copy().transpose()).transpose() - (
                        mu0 / new_mu) * old_x.copy()

            # Save the old
            old_x = curr_x.copy()
            mu0 = mu1
            mu1 = new_mu

        return x